# Q6. Implement Q-Learning for FrozenLake.
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, epsilon = 0.8, 0.95, 0.1

for episode in range(2000):
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state]) if np.random.random() > epsilon else env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
        done = terminated or truncated

print("Learned Q-table:\n", np.round(q_table, 3))
env.close()
