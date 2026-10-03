# Q35. Comparative report summarizing Reinforcement Learning algorithms.
report = """
COMPARATIVE REPORT: REINFORCEMENT LEARNING ALGORITHMS

1. Objective
The experiments study tabular Q-Learning and Deep Q-Network (DQN) methods.

2. Q-Learning
Q-Learning stores action values in a Q-table. It is simple and effective for
small discrete state/action spaces such as deterministic FrozenLake and Grid World.

3. DQN
DQN replaces the Q-table with a neural network. Experience replay and a target
network improve training stability. It is suitable for larger or continuous
observation spaces such as CartPole.

4. Performance Measures
The experiments use episode reward, cumulative reward, success rate, convergence
behavior, training time, and sensitivity to hyperparameters.

5. Hyperparameters
Learning rate (alpha) affects the size of Q-value updates. Gamma controls the
importance of future rewards. Epsilon controls exploration versus exploitation.
DQN additionally depends on batch size, replay-buffer size, optimizer learning
rate, target-network update frequency, and network architecture.

6. Findings to Record
- Average reward after training
- Success rate
- Number of episodes required for stable performance
- Training time
- Effect of replay memory
- Effect of target network
- Effect of learning rate, gamma, and epsilon

7. Conclusion
Tabular Q-Learning is computationally lightweight and interpretable for small
discrete environments. DQN requires more computation and tuning but can handle
high-dimensional observations where a Q-table is impractical.

Note: Enter measured values from Questions 10, 23-34 rather than inventing
experimental results.
"""
print(report)
