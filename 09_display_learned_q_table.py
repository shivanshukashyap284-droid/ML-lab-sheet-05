# Q9. Display the learned Q-table after training.
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((16, 4))
alpha, gamma, epsilon = 0.8, 0.95, 0.1

for _ in range(3000):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.argmax(q[s]) if np.random.rand() > epsilon else env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        q[s, a] += alpha * (r + gamma * np.max(q[ns]) - q[s, a])
        s = ns
        done = term or trunc

actions = ["LEFT", "DOWN", "RIGHT", "UP"]
print("State -> best action -> Q-values")
for s in range(16):
    print(s, "->", actions[np.argmax(q[s])], np.round(q[s], 3))
env.close()
