# Q2. Create and execute a simple Reinforcement Learning environment.
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
print("Initial observation:", obs)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()
