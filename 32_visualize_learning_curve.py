# Q32. Visualize the learning curve of an RL agent.
import numpy as np
import matplotlib.pyplot as plt

rewards=np.random.default_rng(1).binomial(1,0.5,300)
moving=np.convolve(rewards,np.ones(25)/25,mode="valid")
plt.plot(rewards,alpha=.25,label="Episode reward")
plt.plot(range(24,300),moving,label="25-episode average")
plt.xlabel("Episode"); plt.ylabel("Reward")
plt.title("RL Learning Curve")
plt.legend(); plt.show()
