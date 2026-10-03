# Q33. Compare training time and convergence of Q-Learning and DQN.
import time, numpy as np

start=time.perf_counter()
q=np.zeros((16,4))
for _ in range(100000):
    s=np.random.randint(16); a=np.random.randint(4); ns=np.random.randint(16); r=np.random.rand()
    q[s,a]+=.8*(r+.95*np.max(q[ns])-q[s,a])
q_time=time.perf_counter()-start
print("Tabular Q-Learning benchmark time:",round(q_time,4),"seconds")

print("DQN timing should be measured by running Q23 and recording total training time.")
print("Convergence can be compared using moving-average reward and change in loss/Q-values.")
