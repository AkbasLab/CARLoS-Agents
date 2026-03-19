import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rlagent.carlos_gym_env import CarlosGymEnv


# =============================================================================
# CURRICULUM STAGES  (identical structure to ppo_training.py)
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


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
#
# SAC is an off-policy actor-critic algorithm.  Its hyperparameters differ
# substantially from PPO because it uses a replay buffer instead of a rollout
# buffer, and it updates the policy from randomly sampled past transitions
# rather than from the most recent rollout.
#
# learning_rate = 3e-4
#   Standard Adam LR for SAC actor and critic networks.  Original used 1e-4
#   which is conservative and slows convergence unnecessarily for this
#   environment.  3e-4 is the SAC paper default and SB3 default.
#
# buffer_size = 300_000
#   Replay buffer capacity.  Must be large enough to hold diverse experiences
#   across all curriculum conditions.  300k matches total training timesteps
#   so the buffer never discards experience from earlier (easier) stages —
#   this is important because easy-stage experience anchors stable lane-keeping
#   behaviour as harder conditions are introduced.
#   Original had no explicit buffer_size (SB3 default is 1_000_000 which is
#   excessive for 300k total steps and wastes memory).
#
# learning_starts = 2_000
#   Number of random-action steps before any gradient updates begin.
#   This pre-fills the replay buffer with diverse transitions so the first
#   update batch is not highly correlated.  2000 steps ≈ 10 full episodes,
#   which is enough to see the full range of lane/obstacle conditions in
#   Stage 1 before learning starts.
#   Original had no explicit value (SB3 default is 100 — too few to populate
#   a meaningful buffer before updates begin).
#
# batch_size = 256
#   Mini-batch size for each gradient update.  SAC updates every step
#   (train_freq=1), so batch_size controls how much of the replay buffer
#   is used each update.  256 is the SAC paper default; larger than the
#   original 128 to reduce variance given the diverse curriculum buffer.
#
# tau = 0.005
#   Soft target network update coefficient.  Kept from original.
#   Standard value from SAC paper; smaller tau = more stable but slower
#   target tracking.
#
# gamma = 0.98
#   Discount factor.  Matched to PPO and DDPG for consistent comparison
#   across the three algorithms in the paper.  Original used 0.99 — changed
#   to 0.98 so all three algorithms discount future rewards identically.
#
# ent_coef = 'auto'
#   SAC's entropy coefficient is automatically tuned to match a target
#   entropy equal to -dim(action_space) = -2.  This is the key advantage
#   of SAC over DDPG — it self-regulates exploration without manual tuning.
#   Kept from original.
#
# train_freq = 1
#   Update the policy after every environment step.  SAC default.
#   Off-policy algorithms can update more frequently than on-policy ones
#   because they reuse past experience from the replay buffer.
#
# gradient_steps = 4
#   Number of gradient steps per environment step.
#   Increased from 1 because episodes are very short (~5 steps early in
#   training), so the ratio of gradient updates to environment steps is
#   otherwise too low for SAC to learn quickly.  With 4 gradient steps per
#   env step SAC extracts more signal from each transition without needing
#   more environment interactions.  Monitor critic_loss — if it diverges,
#   reduce back to 2.
#
# net_arch = [256, 256]
#   Two hidden layers of 256 units for both actor and critic.
#   SAC has two Q-networks (critics) so the network is slightly larger
#   than PPO's [128, 128] to give each critic enough capacity.
#   Original had no explicit net_arch (SB3 default is [256, 256] for SAC).

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SAC_HYPERPARAMS = dict(
    policy          = 'MlpPolicy',
    learning_rate   = 3e-4,
    buffer_size     = 300_000,
    learning_starts = 500,      # reduced from 2000: with ~5-step episodes,
                                # 2000 steps = 400 crash episodes filling the
                                # buffer before any learning begins.
                                # 500 steps (~100 episodes) is enough to seed
                                # the buffer with diverse starting states.
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.98,
    train_freq      = 1,
    gradient_steps  = 4,        # increased from 1: short episodes mean few
                                # env steps per episode, so more gradient
                                # steps per step compensates and speeds learning.
    ent_coef        = 'auto',
    target_entropy  = -1.0,     # override SAC default of -dim(action)=-2.
                                # With rewards in [-2, +2], the default target
                                # entropy of -2 makes the entropy bonus dominate
                                # the policy objective.  -1.0 reduces entropy
                                # pressure so the policy signal drives learning.
    policy_kwargs   = dict(net_arch=[256, 256]),
    verbose         = 1,
    tensorboard_log = './sac_tensorboard/',
    device          = DEVICE,
)


# =============================================================================
# CURRICULUM ENVIRONMENT  (identical to ppo_training.py)
# =============================================================================
# reset() reinitialises CarlosGymEnv.__init__ in-place with new random
# parameters each episode.  This avoids calling load_lane_from_file()
# manually and is the same pattern used in ppo_training.py.

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
# Important SAC-specific consideration: stage transitions
# -------------------------------------------------------
# PPO is on-policy — it discards all rollout data between updates, so
# swapping the environment with model.set_env() between stages is safe.
#
# SAC is off-policy — its replay buffer is tied to the environment instance
# it was created with.  Calling model.set_env() on a new vec_env would
# mismatch the buffer's stored observations with the new environment's
# observation space (even though the space is identical, SB3 internally
# checks the env reference).
#
# Fix: at each stage transition, save the model weights to disk, create a
# fresh SAC model with the new environment, then load the saved weights
# back in.  This gives the new model the learned policy while letting it
# build a fresh replay buffer for the new stage's condition pool.
# The replay buffer is intentionally NOT transferred between stages so
# that the agent explores the new (harder) conditions before reusing old
# easy-stage experience.

def train_curriculum(model_save_path='sac_carlos_agent'):
    """
    Train SAC through three curriculum stages.

    At each stage transition the policy weights are preserved but the
    replay buffer is reset so the agent explores the new harder conditions
    before drawing on old experience.
    """
    print()
    print('=' * 68)
    print(f'  SAC CURRICULUM TRAINING  [{DEVICE.upper()}]')
    print(f'  Total timesteps : {TOTAL_TIMESTEPS:,}  across {len(CURRICULUM_STAGES)} stages')
    print(f'  Reward pool     : {len(REWARD_POOL)} configs randomised per episode')
    print('=' * 68)

    _STAGE_WEIGHTS = '_sac_stage_weights_tmp'   # temp file for weight transfer

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

        # SAC does not use make_vec_env — it works with a single env directly
        # (SB3 SAC wraps it internally).  Passing n_envs>1 is not supported
        # for off-policy algorithms in standard SB3.
        env = CurriculumEnv(
            lane_pool   = stage['lane_widths'],
            obs_pool    = stage['obs_counts'],
            reward_pool = REWARD_POOL,
        )

        if model is None:
            # First stage: create model from scratch
            model = SAC(env=env, **SAC_HYPERPARAMS)
        else:
            # Subsequent stages: preserve policy weights, fresh replay buffer
            # Save weights from the previous stage model
            model.save(_STAGE_WEIGHTS)
            # Create a fresh model with the new environment
            model = SAC(env=env, **SAC_HYPERPARAMS)
            # Load only the policy weights (not the replay buffer)
            model.set_parameters(_STAGE_WEIGHTS)
            print(f'    Weights loaded from previous stage.')

        os.makedirs('./checkpoints', exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq   = 50_000,
            save_path   = './checkpoints/',
            name_prefix = f'sac_{stage["name"]}',
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
    Train one SAC model under a fixed reward config with randomised geometry.
    Used by run_reward_shaping_sweep() in the test file.
    """
    print(f'  Training {train_timesteps:,} steps  weights: {reward_weights}')

    env = CurriculumEnv(
        lane_pool   = [10, 11, 12, 14],
        obs_pool    = [4, 6, 8, 10],
        reward_pool = [reward_weights],   # fixed config, no randomisation
    )

    model = SAC(env=env, **SAC_HYPERPARAMS)
    model.learn(total_timesteps=train_timesteps)
    model.save(model_save_path)
    env.close()
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    train_curriculum(model_save_path='sac_carlos_agent')


if __name__ == '__main__':
    main()