import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from rlagent.carlos_gym_env import CarlosGymEnv
import itertools

"""
This script performs hyperparameter tuning for the SAC model using grid search.
It evaluates each combination and prints the best hyperparameters based on mean reward.
"""

def main():
    import torch
    env = CarlosGymEnv()

    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define hyperparameter grid
    learning_rates = [3e-4, 1e-4, 1e-5]
    batch_sizes = [64, 128, 256]
    gammas = [0.98, 0.99, 1]
    taus = [0.005, 0.01]
    ent_coefs = ["auto", 0.1]

    param_grid = list(itertools.product(learning_rates, batch_sizes, gammas, taus, ent_coefs))
    import csv
    results = []
    best_mean_reward = -np.inf
    best_params = None

    for lr, batch_size, gamma, tau, ent_coef in param_grid:
        print(f"Testing: lr={lr}, batch_size={batch_size}, gamma={gamma}, tau={tau}, ent_coef={ent_coef}")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef,
            device=device
        )
        # Train for a small number of steps for quick tuning
        model.learn(total_timesteps=10000)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=False)
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'gamma': gamma,
            'tau': tau,
            'ent_coef': ent_coef,
            'device': device,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_params = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'gamma': gamma,
                'tau': tau,
                'ent_coef': ent_coef,
                'device': device
            }

    # Write results to CSV
    csv_file = "sac_hyperparam_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'learning_rate', 'batch_size', 'gamma', 'tau', 'ent_coef', 'device', 'mean_reward', 'std_reward'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nAll results saved to {csv_file}")

    print("\nBest hyperparameters:")
    print(best_params)
    print(f"Best mean reward: {best_mean_reward:.2f}")

if __name__ == "__main__":
    main()
