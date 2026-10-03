REINFORCEMENT LEARNING LAB - 35 PYTHON PROGRAMS

Files 01-35 correspond directly to Questions 1-35.
Recommended environment:
    python -m venv .venv
    # Windows Git Bash:
    source .venv/Scripts/activate
    pip install numpy matplotlib gymnasium torch

For CartPole extras:
    pip install "gymnasium[classic-control]"

Run any program:
    python 01_install_configure_gymnasium.py

Important:
- Q23 trains a DQN and can take time depending on CPU/GPU.
- Q24 expects dqn_rewards.npy if you want to plot actual rewards. You can add
  np.save("dqn_rewards.npy", rewards) at the end of Q23.
- Q25/Q30 require dqn_cartpole.pth from Q23 or Q29.
- Q26 explains why direct numerical comparison across FrozenLake and CartPole
  is not scientifically equivalent.
