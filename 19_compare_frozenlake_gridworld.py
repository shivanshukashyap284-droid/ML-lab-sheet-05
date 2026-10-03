# Q19. Compare agent performance in FrozenLake and Grid World.
# This script reports success/reward under the same Q-Learning idea.
import numpy as np
import gymnasium as gym

# FrozenLake
env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4))
for _ in range(3000):
    s,_=env.reset(); d=False
    while not d:
        a=np.argmax(q[s]) if np.random.rand()>.1 else env.action_space.sample()
        ns,r,t,tr,_=env.step(a); q[s,a]+=.8*(r+.95*np.max(q[ns])-q[s,a]); s=ns; d=t or tr
scores=[]
for _ in range(100):
    s,_=env.reset(); total=0; d=False
    while not d:
        s,r,t,tr,_=env.step(np.argmax(q[s])); total+=r; d=t or tr
    scores.append(total)
env.close()
print("FrozenLake success rate:",np.mean(scores)*100,"%")

# Deterministic Grid World
n=5; q=np.zeros((25,4))
def step(s,a):
    r,c=divmod(s,n)
    if a==0:r=max(0,r-1)
    elif a==1:c=min(n-1,c+1)
    elif a==2:r=min(n-1,r+1)
    else:c=max(0,c-1)
    ns=r*n+c; return ns,(10 if ns==24 else -1),ns==24
for _ in range(3000):
    s=0; d=False
    for _ in range(100):
        a=np.argmax(q[s]) if np.random.rand()>.1 else np.random.randint(4)
        ns,r,d=step(s,a); q[s,a]+=.8*(r+.95*np.max(q[ns])-q[s,a]); s=ns
        if d: break
success=0
for _ in range(100):
    s=0
    for _ in range(100):
        s,r,d=step(s,np.argmax(q[s]))
        if d: success+=1; break
print("Grid World success rate:",success,"%")
