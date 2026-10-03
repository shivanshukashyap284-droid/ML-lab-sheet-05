# Q10. Evaluate the trained Q-Learning agent.
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((16, 4))
alpha, gamma, epsilon = 0.8, 0.95, 0.1

for _ in range(4000):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.argmax(q[s]) if np.random.rand() > epsilon else env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        q[s, a] += alpha * (r + gamma*np.max(q[ns]) - q[s,a])
        s, done = ns, term or trunc

rewards = []
for ep in range(100):
    s, _ = env.reset()
    total = 0
    done = False
    while not done:
        a = np.argmax(q[s])
        s, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    rewards.append(total)

print("Episodes:", len(rewards))
print("Average reward:", np.mean(rewards))
print("Success rate:", np.mean(rewards) * 100, "%")
env.close()
