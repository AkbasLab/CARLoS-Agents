from stable_baselines3 import PPO
from rlagent.carlos_gym_env import CarlosGymEnv

def main():
    env = CarlosGymEnv()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        learning_rate=3e-5,   
        n_steps=2048,         
        batch_size=1024,        
        gamma=0.99,           
        clip_range=0.15
    )
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)

    model.save("tuned_ppo_carlos_agent")

    print("Training complete. Model saved as tuned_ppo_carlos_agent.zip")

if __name__ == "__main__":
    main()
