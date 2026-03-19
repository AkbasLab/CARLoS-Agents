import os
import random
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from rlagent.carlos_gym_env import CarlosGymEnv


# =============================================================================
# CURRICULUM STAGES  (identical structure to ddpg_training.py)
# =============================================================================
# Three stages of increasing difficulty.
# Each stage defines which lane widths and obstacle counts are sampled
# randomly at every episode reset.  Weights carry forward across stages.
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

# Action space dimension — used to build action noise.
# CarlosGymEnv action space is [steering, throttle] → 2-dimensional.
ACTION_DIM = 2


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
#
# DDPG is a deterministic off-policy actor-critic algorithm.  Unlike SAC it
# has no entropy term, so all exploration comes entirely from action noise
# injected at decision time.  This makes the noise configuration critical.
#
# learning_rate = 1e-4
#   Separate LR for actor and critic.  Original used 1e-5 which is extremely
#   conservative — at that rate DDPG rarely converges within 300k steps for
#   a continuous control task.  1e-4 is the standard DDPG paper value and
#   gives meaningful policy improvement within the training budget.
#   SAC uses 3e-4 (higher is fine for SAC because entropy regularisation
#   stabilises learning); DDPG without entropy needs the slightly lower 1e-4
#   to avoid actor divergence.
#
# buffer_size = 300_000
#   Same reasoning as SAC: matches total training steps so no curriculum
#   experience is evicted during training.  Original had no explicit value.
#
# learning_starts = 2_000
#   Same reasoning as SAC: pre-fills buffer with ~10 episodes of random
#   experience before any gradient updates begin.  Original had no explicit
#   value (SB3 default is 100 — too few).
#
# batch_size = 256
#   Mini-batch size for each gradient update.  DDPG paper default.
#   Original used 256; kept — appropriate for replay buffer sampling.
#
# tau = 0.005
#   Soft target network update.  Kept from original.  Standard DDPG value.
#
# gamma = 0.98
#   Matched to ddpg and SAC for consistent cross-algorithm comparison.
#   Original used 0.98; kept.
#
# train_freq = 1
#   Update after every environment step.  DDPG default (off-policy).
#
# gradient_steps = 1
#   One gradient step per environment step.  Standard for DDPG.
#   Unlike SAC, DDPG is more sensitive to gradient_steps > 1 because there
#   is no entropy regularisation to stabilise the critic.
#
# action_noise — OrnsteinUhlenbeckActionNoise (OU)
#   DDPG was originally designed with OU noise, which produces temporally
#   correlated noise.  This is important for driving tasks because steering
#   and throttle commands are smooth continuous signals — uncorrelated
#   Gaussian noise (NormalActionNoise) would produce jerky, unrealistic
#   actions that are hard to learn from.
#
#   sigma = 0.2 (original was 0.1)
#   Higher sigma in Stage 1 so the agent explores more aggressively in the
#   easy conditions before the geometry gets tight.  The noise is shared
#   across all stages for simplicity; in a more elaborate setup you would
#   anneal sigma from 0.2 down to 0.05 over training.
#
#   theta = 0.15, dt = 1e-2 — OU process standard parameters from the
#   original DDPG paper.
#
# net_arch = [400, 300]
#   The original DDPG paper used [400, 300] for actor and critic.
#   This is larger than SAC's [256, 256] because DDPG has only one critic
#   (vs SAC's two) so it needs more capacity per network.  The SB3 default
#   for DDPG is [400, 300], matching the paper.

ACTION_NOISE = OrnsteinUhlenbeckActionNoise(
    mean  = np.zeros(ACTION_DIM),
    sigma = 0.2 * np.ones(ACTION_DIM),
    theta = 0.15,
    dt    = 1e-2,
)

DDPG_HYPERPARAMS = dict(
    policy          = 'MlpPolicy',
    learning_rate   = 1e-4,
    buffer_size     = 300_000,
    learning_starts = 500,      # reduced from 2000: with ~5-step episodes,
                                # 2000 steps = 400 random-action crash episodes
                                # before a single gradient update fires.
                                # 500 steps (~100 episodes) seeds the buffer
                                # without starving the agent of learning signal.
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.98,
    train_freq      = 1,
    gradient_steps  = 4,        # increased from 1: short episodes mean very
                                # few env steps per episode, so more gradient
                                # steps per step compensates. DDPG is more
                                # sensitive than SAC here (no entropy
                                # regularisation), so monitor critic_loss —
                                # reduce to 2 if it diverges.
    action_noise    = ACTION_NOISE,
    policy_kwargs   = dict(net_arch=[400, 300]),
    verbose         = 1,
    tensorboard_log = './ddpg_tensorboard/',
)


# =============================================================================
# CURRICULUM ENVIRONMENT  (identical to ppo_training.py and sac_training.py)
# =============================================================================
# reset() reinitialises CarlosGymEnv.__init__ in-place with new random
# parameters each episode.  Same pattern as PPO and SAC training files.

class CurriculumEnv(CarlosGymEnv):
    """
    CarlosGymEnv that randomises lane width, obstacle count, and reward
    weights at every episode reset by reinitialising from scratch.
    """

    def __init__(self, lane_pool, obs_pool, reward_pool):
        self._lane_pool   = lane_pool
        self._obs_pool    = obs_pool
        self._reward_pool = reward_pool

        super().__init__(
            lane_width_ft  = random.choice(lane_pool),
            num_obstacles  = random.choice(obs_pool),
            reward_weights = random.choice(reward_pool),
        )

    def reset(self, seed=None, options=None):
        lw = random.choice(self._lane_pool)
        no = random.choice(self._obs_pool)
        rw = random.choice(self._reward_pool)

        CarlosGymEnv.__init__(
            self,
            lane_width_ft  = lw,
            num_obstacles  = no,
            reward_weights = rw,
        )
        return super().reset(seed=seed, options=options)


# =============================================================================
# CURRICULUM TRAINING
# =============================================================================
#
# Stage transition strategy (same as SAC, different from PPO)
# -----------------------------------------------------------
# DDPG is off-policy like SAC, so the same replay-buffer mismatch issue
# applies when changing environments between stages.  The same fix is used:
# save policy weights at the end of each stage, create a fresh DDPG model
# with the new environment, reload the weights.  The replay buffer is reset
# intentionally so the agent explores the new harder conditions fresh.
#
# Action noise reset between stages
# ----------------------------------
# OrnsteinUhlenbeck noise is stateful — it has an internal state that drifts
# over time.  When the environment changes at a stage boundary, the OU state
# is also reset by creating a new ACTION_NOISE instance, so the noise process
# starts fresh and is not biased by the previous stage's drift.

def train_curriculum(model_save_path='ddpg_carlos_agent'):
    """
    Train DDPG through three curriculum stages.

    At each stage transition policy weights are preserved, replay buffer
    and OU noise state are reset for the new condition pool.
    """
    print()
    print('=' * 68)
    print('  DDPG CURRICULUM TRAINING')
    print(f'  Total timesteps : {TOTAL_TIMESTEPS:,}  across {len(CURRICULUM_STAGES)} stages')
    print(f'  Reward pool     : {len(REWARD_POOL)} configs randomised per episode')
    print(f'  Action noise    : OrnsteinUhlenbeck  sigma=0.2  theta=0.15')
    print('=' * 68)

    _STAGE_WEIGHTS = '_ddpg_stage_weights_tmp'

    model = None

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        print()
        print(f'  Stage {stage_idx + 1}/{len(CURRICULUM_STAGES)}: {stage["name"]}')
        print(f'    Lane widths : {stage["lane_widths"]} ft')
        print(f'    Obstacles   : {stage["obs_counts"]}')
        print(f'    Timesteps   : {stage["timesteps"]:,}')

        env = CurriculumEnv(
            lane_pool   = stage['lane_widths'],
            obs_pool    = stage['obs_counts'],
            reward_pool = REWARD_POOL,
        )

        # Fresh OU noise instance at each stage so the noise state is not
        # biased by drift accumulated in the previous stage.
        stage_noise = OrnsteinUhlenbeckActionNoise(
            mean  = np.zeros(ACTION_DIM),
            sigma = 0.2 * np.ones(ACTION_DIM),
            theta = 0.15,
            dt    = 1e-2,
        )
        stage_hyperparams = {**DDPG_HYPERPARAMS, 'action_noise': stage_noise}

        if model is None:
            model = DDPG(env=env, **stage_hyperparams)
        else:
            # Save weights from previous stage, create fresh model, reload weights
            model.save(_STAGE_WEIGHTS)
            model = DDPG(env=env, **stage_hyperparams)
            model.set_parameters(_STAGE_WEIGHTS)
            print(f'    Weights loaded from previous stage.')

        os.makedirs('./checkpoints', exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq   = 50_000,
            save_path   = './checkpoints/',
            name_prefix = f'ddpg_{stage["name"]}',
        )

        model.learn(
            total_timesteps     = stage['timesteps'],
            callback            = checkpoint_cb,
            reset_num_timesteps = (stage_idx == 0),
        )

        stage_path = f'{model_save_path}_{stage["name"]}'
        model.save(stage_path)
        print(f'    Saved: {stage_path}.zip')
        env.close()

    # Clean up temporary weight file
    tmp = f'{_STAGE_WEIGHTS}.zip'
    if os.path.exists(tmp):
        os.remove(tmp)

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
    Train one DDPG model under a fixed reward config with randomised geometry.
    Used by run_reward_shaping_sweep() in the test file.
    """
    print(f'  Training {train_timesteps:,} steps  weights: {reward_weights}')

    env = CurriculumEnv(
        lane_pool   = [10, 11, 12, 14],
        obs_pool    = [4, 6, 8, 10],
        reward_pool = [reward_weights],   # fixed config, no randomisation
    )

    model = DDPG(env=env, **DDPG_HYPERPARAMS)
    model.learn(total_timesteps=train_timesteps)
    model.save(model_save_path)
    env.close()
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    train_curriculum(model_save_path='ddpg_carlos_agent')


if __name__ == '__main__':
    main()