from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from rlagent.carlos_gym_env import CarlosGymEnv
import numpy as np

def main():
    env = CarlosGymEnv()

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ddpg_tensorboard/", 
        learning_rate = 1e-3,
        batch_size = 128,
        gamma = 0.99,
        tau = 0.005, 
        action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]),
                                 sigma=0.1 * np.ones(env.action_space.shape[0]))
    )

    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)

    model.save("ddpg_carlos_agent")

    print("Training complete. Model saved as ddpg_carlos_agent.zip")

if __name__ == "__main__":
    main()