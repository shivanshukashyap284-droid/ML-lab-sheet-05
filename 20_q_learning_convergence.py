# Q20. Analyze convergence behavior of Q-Learning.
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4))
max_changes=[]; episode_rewards=[]
for ep in range(3000):
    s,_=env.reset(); d=False; change=0; total=0
    while not d:
        a=np.argmax(q[s]) if np.random.rand()>.1 else env.action_space.sample()
        ns,r,t,tr,_=env.step(a)
        old=q[s,a]; q[s,a]+=.8*(r+.95*np.max(q[ns])-q[s,a])
        change=max(change,abs(q[s,a]-old)); total+=r; s=ns; d=t or tr
    max_changes.append(change); episode_rewards.append(total)
env.close()
plt.plot(max_changes)
plt.xlabel("Episode"); plt.ylabel("Maximum Q-value update")
plt.title("Q-Learning Convergence")
plt.show()
print("Last 100 average update:",np.mean(max_changes[-100:]))
