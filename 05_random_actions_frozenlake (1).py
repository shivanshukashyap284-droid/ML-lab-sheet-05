# Q5. Simulate random actions in FrozenLake.
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
episodes = 5

for ep in range(episodes):
    state, info = env.reset(seed=ep)
    total_reward = 0
    done = False
    print(f"\nEpisode {ep+1}")
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        print(f"action={action}, state={state}, reward={reward}")
    print("Total reward:", total_reward)

env.close()
