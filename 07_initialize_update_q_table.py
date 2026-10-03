# Q7. Initialize and update the Q-table during training.
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((env.observation_space.n, env.action_space.n))
print("Initial Q-table:\n", q)

alpha, gamma = 0.8, 0.95
state, _ = env.reset()
action = env.action_space.sample()
next_state, reward, terminated, truncated, _ = env.step(action)

old = q[state, action]
target = reward + gamma * np.max(q[next_state])
q[state, action] = old + alpha * (target - old)

print("\nState:", state, "Action:", action, "Reward:", reward)
print("Updated Q-table:\n", q)
env.close()
