# Q11. Plot cumulative rewards during training.
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((16,4))
alpha, gamma, epsilon = 0.8, 0.95, 0.1
rewards = []

for _ in range(3000):
    s, _ = env.reset()
    total, done = 0, False
    while not done:
        a = np.argmax(q[s]) if np.random.rand() > epsilon else env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        q[s,a] += alpha*(r + gamma*np.max(q[ns]) - q[s,a])
        s, total, done = ns, total+r, term or trunc
    rewards.append(total)

cumulative = np.cumsum(rewards)
plt.plot(cumulative)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards - Q-Learning")
plt.show()
env.close()
