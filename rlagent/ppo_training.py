import os
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rlagent.carlos_gym_env import CarlosGymEnv


# =============================================================================
# CURRICULUM STAGES
# =============================================================================
# Three stages of increasing difficulty.
# Each stage defines which lane widths and obstacle counts are sampled
# randomly at every episode reset.  The model carries its weights forward
# across stages — it builds on what it learned rather than restarting.
#
# Held-out condition: (10 ft, 10 obstacles) — never seen during training.
# Reserved as the hardest stress-test condition in the evaluation sweep.

CURRICULUM_STAGES = [
    {
        'name'       : 'Stage1_Easy',
        'timesteps'  : 80_000,
        'lane_widths': [14, 12],
        'obs_counts' : [4, 6],
    },
    {
        'name'       : 'Stage2_Medium',
        'timesteps'  : 100_000,
        'lane_widths': [12, 11],
        'obs_counts' : [6, 8],
    },
    {
        'name'       : 'Stage3_Hard',
        'timesteps'  : 120_000,
        'lane_widths': [14, 12, 11, 10],
        'obs_counts' : [4, 6, 8, 10],
    },
]

TOTAL_TIMESTEPS = sum(s['timesteps'] for s in CURRICULUM_STAGES)  # 300_000

# Reward weights randomised per episode so the policy is not over-fitted
# to one incentive structure.  Matches the three configs in the eval sweep.
REWARD_POOL = [
    {'in_lane':  1.0, 'in_motion':  0.5, 'collision': -10.0, 'out_lane':  -5.0},
    {'in_lane':  1.0, 'in_motion':  1.0, 'collision':  -5.0, 'out_lane':  -2.0},
    {'in_lane':  2.0, 'in_motion':  0.3, 'collision': -50.0, 'out_lane': -20.0},
]


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# n_steps=2048   — ~10 complete episodes per update (200 steps each).
#                  Original 512 gave only ~2.5 episodes, too noisy.
# batch_size=128 — divides n_steps evenly (2048/128=16 mini-batches).
# clip_range=0.2 — standard PPO paper value; original 0.3 was too aggressive.
# ent_coef=0.01  — entropy bonus to maintain exploration; was missing (0.0).
# lr schedule    — decays 3e-4 → 5e-5 to stabilise late-stage convergence.
# net_arch=[128,128] — slightly larger than original [64,64] for 16 conditions.

def linear_lr_schedule(initial_lr, final_lr):
    def schedule(progress_remaining):
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


PPO_HYPERPARAMS = dict(
    policy          = 'MlpPolicy',
    learning_rate   = linear_lr_schedule(3e-4, 5e-5),
    n_steps         = 2048,
    batch_size      = 128,
    n_epochs        = 10,
    gamma           = 0.98,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    policy_kwargs   = dict(net_arch=[128, 128]),
    verbose         = 1,
    tensorboard_log = './ppo_tensorboard/',
)


# =============================================================================
# CURRICULUM ENVIRONMENT
# =============================================================================
# The key insight: instead of rebuilding the lane geometry inside reset(),
# which requires calling load_lane_from_file() again and caused the crash,
# we create a brand-new CarlosGymEnv instance at every reset().
#
# CarlosGymEnv.__init__ already handles all setup (lane, obstacles, sim).
# Recreating the object is the cleanest way to get a fresh randomised
# condition without reimplementing any of that logic here.
#
# SB3 calls env.reset() at the end of every episode.  Each call here
# samples new lane_width, num_obstacles, and reward_weights, then
# delegates to a freshly constructed CarlosGymEnv.

class CurriculumEnv(CarlosGymEnv):
    """
    CarlosGymEnv that randomises lane width, obstacle count, and reward
    weights at every episode reset by reinitialising from scratch.

    Parameters
    ----------
    lane_pool   : list[int]   — lane widths (ft) to sample from.
    obs_pool    : list[int]   — obstacle counts to sample from.
    reward_pool : list[dict]  — reward weight dicts to sample from.
    """

    def __init__(self, lane_pool, obs_pool, reward_pool):
        self._lane_pool   = lane_pool
        self._obs_pool    = obs_pool
        self._reward_pool = reward_pool

        # Initialise with one random sample so the object is valid from the start
        super().__init__(
            lane_width_ft  = random.choice(lane_pool),
            num_obstacles  = random.choice(obs_pool),
            reward_weights = random.choice(reward_pool),
        )

    def reset(self, seed=None, options=None):
        """
        At the start of every new episode, pick a fresh random condition
        and reinitialise the entire environment from scratch.

        Calling CarlosGymEnv.__init__ rebuilds the lane, places new
        obstacles, and wires up a fresh SimulationRL — exactly what
        __init__ already does reliably.  This avoids any need to call
        load_lane_from_file() ourselves or patch internal sim state.
        """
        lw = random.choice(self._lane_pool)
        no = random.choice(self._obs_pool)
        rw = random.choice(self._reward_pool)

        # Reinitialise in-place: rebuilds lane, obstacles, sim cleanly
        CarlosGymEnv.__init__(
            self,
            lane_width_ft  = lw,
            num_obstacles  = no,
            reward_weights = rw,
        )

        # Now call the standard reset (resets step counter, calls sim_random_reset)
        return super().reset(seed=seed, options=options)


# =============================================================================
# CURRICULUM TRAINING
# =============================================================================

def train_curriculum(model_save_path='ppo_carlos_agent'):
    """
    Train PPO through three curriculum stages.

    Each stage creates a new CurriculumEnv with a harder pool of conditions.
    The model weights carry forward across stages.
    A checkpoint is saved after each stage plus every 50k steps.
    """
    print()
    print('=' * 68)
    print('  PPO CURRICULUM TRAINING')
    print(f'  Total timesteps : {TOTAL_TIMESTEPS:,}  across {len(CURRICULUM_STAGES)} stages')
    print(f'  Reward pool     : {len(REWARD_POOL)} configs randomised per episode')
    print('=' * 68)

    model = None

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        print()
        print(f'  Stage {stage_idx + 1}/{len(CURRICULUM_STAGES)}: {stage["name"]}')
        print(f'    Lane widths : {stage["lane_widths"]} ft')
        print(f'    Obstacles   : {stage["obs_counts"]}')
        print(f'    Timesteps   : {stage["timesteps"]:,}')

        def make_env(lane_pool=stage['lane_widths'],
                     obs_pool=stage['obs_counts'],
                     reward_pool=REWARD_POOL):
            return CurriculumEnv(lane_pool, obs_pool, reward_pool)

        vec_env = make_vec_env(make_env, n_envs=1)

        if model is None:
            model = PPO(env=vec_env, **PPO_HYPERPARAMS)
        else:
            model.set_env(vec_env)

        os.makedirs('./checkpoints', exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq   = 50_000,
            save_path   = './checkpoints/',
            name_prefix = f'ppo_{stage["name"]}',
        )

        model.learn(
            total_timesteps     = stage['timesteps'],
            callback            = checkpoint_cb,
            reset_num_timesteps = (stage_idx == 0),
        )

        stage_path = f'{model_save_path}_{stage["name"]}'
        model.save(stage_path)
        print(f'    Saved: {stage_path}.zip')
        vec_env.close()

    model.save(model_save_path)
    print()
    print(f'  Final model saved as {model_save_path}.zip')
    print('=' * 68)
    return model


# =============================================================================
# SINGLE-CONFIG TRAINING  (used by reward shaping sweep in final_ppo_test.py)
# =============================================================================

def train_single_config(reward_weights, model_save_path, train_timesteps=300_000):
    """
    Train one PPO model under a fixed reward config with randomised geometry.

    Used by run_reward_shaping_sweep() in final_ppo_test.py.  All three
    reward-config models see the same geometry distribution so differences
    in evaluation metrics are attributable to the reward signal alone.
    """
    print(f'  Training {train_timesteps:,} steps  weights: {reward_weights}')

    def make_env(rw=reward_weights):
        return CarlosGymEnv(
            lane_width_ft  = random.choice([10, 11, 12, 14]),
            num_obstacles  = random.choice([4, 6, 8, 10]),
            reward_weights = rw,
        )

    vec_env = make_vec_env(make_env, n_envs=1)
    model   = PPO(env=vec_env, **PPO_HYPERPARAMS)
    model.learn(total_timesteps=train_timesteps)
    model.save(model_save_path)
    vec_env.close()
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    train_curriculum(model_save_path='ppo_carlos_agent')


if __name__ == '__main__':
    main()