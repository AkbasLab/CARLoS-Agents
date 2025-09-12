import numpy as np
from stable_baselines3 import DDPG
from rlagent.carlos_gym_env import CarlosGymEnv

def is_outside_lane(info):
    return not info.get('in_lane', True)

def has_collision(info):
    return info.get('collision', False)

def distance_to_lane_edge(info):
    return info.get('safety_margin', 0.0)

def main():
    env = CarlosGymEnv()
    model = DDPG.load("ddpg_carlos_agent")
    num_episodes = 50
    max_steps = env.max_steps

    lane_violation_counts = 0
    total_steps = 0
    collision_episodes = 0
    failure_steps = []
    safety_margins_all = []
    steering_records = []
    acceleration_records = []
    total_reward=0

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        episode_steps = 0
        episode_failed = False

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs, reward, terminated, truncated, info)

            total_steps += 1
            episode_steps += 1

            if is_outside_lane(info):
                lane_violation_counts += 1
                if not episode_failed:
                    failure_steps.append(episode_steps)
                    episode_failed = True

            if has_collision(info):
                if not episode_failed:
                    failure_steps.append(episode_steps)
                    episode_failed = True
                collision_episodes += 1

            safety_margins_all.append(distance_to_lane_edge(info))

            steering_records.append(action[0])
            acceleration_records.append(action[1])
            total_reward+=reward

        #env.render()

    lane_violation_rate = (lane_violation_counts / total_steps) * 100 if total_steps > 0 else 0.0
    collision_rate = (collision_episodes / num_episodes) * 100 if num_episodes > 0 else 0.0
    avg_time_to_failure = np.mean(failure_steps) if failure_steps else max_steps
    mean_safety_margin = np.mean(safety_margins_all) if safety_margins_all else 0.0
    steering_var = np.var(steering_records) if steering_records else 0.0
    acceleration_var = np.var(acceleration_records) if acceleration_records else 0.0
    control_smoothness = (steering_var + acceleration_var) / 2

    print(f"Safety Metrics after {num_episodes} episodes:")
    print(f"Lane Violation Rate: {lane_violation_rate:.2f}%")
    print(f"Collision Rate: {collision_rate:.2f}%")
    print(f"Average Time to Failure: {avg_time_to_failure:.2f} steps")
    print(f"Mean Safety Margin: {mean_safety_margin:.3f} meters")
    print(f"Control Smoothness (Variance): {control_smoothness:.5f}")

if __name__ == "__main__":
    main()
