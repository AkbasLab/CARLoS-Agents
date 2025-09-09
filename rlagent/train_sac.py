from stable_baselines3 import SAC
from rlagent.carlos_gym_env import CarlosGymEnv

def main():
    env = CarlosGymEnv()

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./sac_tensorboard/",
        learning_rate = 3e-4,
        ent_coef = "auto",      
        gamma = 0.99,           
        batch_size = 256,      
        tau = 0.005
    )

    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)

    model.save("sac_carlos_agent")

    print("Training complete. Model saved as sac_carlos_agent.zip")

if __name__ == "__main__":
    main()