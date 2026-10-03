# Q31. Compare cumulative rewards from different RL algorithms.
import numpy as np
import matplotlib.pyplot as plt

q_learning=np.array([0,1,0,1,1,0,1,1,1,1])
dqn=np.array([1,1,2,1,3,2,4,5,5,6])
plt.plot(np.cumsum(q_learning),label="Q-Learning")
plt.plot(np.cumsum(dqn),label="DQN")
plt.xlabel("Episode"); plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")
plt.legend(); plt.show()
print("For a real experiment, replace the example arrays with rewards recorded from Q-Learning and DQN.")
