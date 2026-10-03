# Q24. Plot episode-wise DQN rewards.
# If Q23 has been run, save rewards to a file or replace this list with its rewards.
import numpy as np
import matplotlib.pyplot as plt

# Demonstration/reproducible placeholder curve for plotting structure.
# For actual experiment, paste/load the rewards list produced by Q23.
rewards=np.load("dqn_rewards.npy") if __import__("os").path.exists("dqn_rewards.npy") else []
if len(rewards)==0:
    print("Run Q23 and save rewards with: np.save('dqn_rewards.npy', rewards)")
else:
    plt.plot(rewards)
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("DQN Episode-wise Reward")
    plt.show()
