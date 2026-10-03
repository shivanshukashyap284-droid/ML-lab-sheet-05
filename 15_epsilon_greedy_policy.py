# Q15. Implement epsilon-greedy action selection.
import numpy as np

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))  # exploration
    return int(np.argmax(q_values))               # exploitation

q_values = np.array([0.1, 0.8, 0.3, 0.2])
for epsilon in [1.0, 0.5, 0.1, 0.0]:
    choices = [epsilon_greedy(q_values, epsilon) for _ in range(20)]
    print(f"Epsilon={epsilon}: actions={choices}")
