# Q17. Train a Q-Learning agent in custom Grid World.
import numpy as np

class GridWorld:
    def __init__(self,n=5): self.n=n; self.start=(0,0); self.goal=(n-1,n-1)
    def reset(self): self.s=self.start; return self.s
    def step(self,a):
        r,c=self.s
        if a==0:r=max(0,r-1)
        elif a==1:c=min(self.n-1,c+1)
        elif a==2:r=min(self.n-1,r+1)
        else:c=max(0,c-1)
        self.s=(r,c); d=self.s==self.goal
        return self.s,(10 if d else -1),d
    def idx(self,s): return s[0]*self.n+s[1]

env=GridWorld(); q=np.zeros((25,4))
for ep in range(5000):
    s=env.reset(); done=False
    for _ in range(100):
        a=np.argmax(q[env.idx(s)]) if np.random.rand()>.1 else np.random.randint(4)
        ns,r,done=env.step(a)
        q[env.idx(s),a]+=.8*(r+.95*np.max(q[env.idx(ns)])-q[env.idx(s),a])
        s=ns
        if done: break
print("Training completed.")
print("Q-table:\n",np.round(q,2))
