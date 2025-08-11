from stable_baselines3 import PPO
from rlagent.carlos_gym_env import CarlosGymEnv

def main():
    env = CarlosGymEnv()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/", 
    )
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps)

    model.save("ppo_carlos_agent")

    print("Training complete. Model saved as ppo_carlos_agent.zip")

if __name__ == "__main__":
    main()
