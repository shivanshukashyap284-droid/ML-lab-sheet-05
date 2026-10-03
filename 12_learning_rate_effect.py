# Q12. Study the effect of different learning rates.
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def train(alpha, episodes=2000):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    q = np.zeros((16,4))
    rewards = []
    for _ in range(episodes):
        s,_ = env.reset()
        total,done = 0,False
        while not done:
            a = np.argmax(q[s]) if np.random.rand() > .1 else env.action_space.sample()
            ns,r,term,trunc,_ = env.step(a)
            q[s,a] += alpha*(r + .95*np.max(q[ns]) - q[s,a])
            s,total,done = ns,total+r,term or trunc
        rewards.append(total)
    env.close()
    return np.convolve(rewards, np.ones(100)/100, mode="valid")

for alpha in [0.1, 0.5, 0.9]:
    plt.plot(train(alpha), label=f"alpha={alpha}")
plt.xlabel("Episode")
plt.ylabel("100-Episode Moving Average Reward")
plt.title("Effect of Learning Rate")
plt.legend()
plt.show()
