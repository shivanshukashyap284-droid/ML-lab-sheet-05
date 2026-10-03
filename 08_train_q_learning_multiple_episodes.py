# Q8. Train an agent for multiple episodes using Q-Learning.
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, epsilon = 0.8, 0.95, 0.1
episodes = 5000

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q[state]) if np.random.rand() > epsilon else env.action_space.sample()
        ns, r, term, trunc, _ = env.step(action)
        q[state, action] += alpha * (r + gamma * np.max(q[ns]) - q[state, action])
        state = ns
        done = term or trunc

print(f"Training completed for {episodes} episodes.")
print("Final Q-table:\n", np.round(q, 3))
env.close()
