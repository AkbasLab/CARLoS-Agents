from rlagent.carlos_gym_env import CarlosGymEnv

env = CarlosGymEnv()

num_episodes = 10  
for episode in range(num_episodes):
    obs, info = env.reset()
    print(f"Episode {episode + 1} initial observation:", obs)
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)
        env.render() 

#rl_env\Scripts\activate