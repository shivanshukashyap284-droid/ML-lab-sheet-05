# Q1. Install and configure Gymnasium.
# Run in terminal first if needed:
# pip install gymnasium
import gymnasium as gym
print("Gymnasium version:", gym.__version__)
env = gym.make("FrozenLake-v1", is_slippery=False)
print("Environment created successfully:", env)
env.close()
