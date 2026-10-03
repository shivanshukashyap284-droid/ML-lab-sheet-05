# Q16. Design a simple Grid World environment.
import numpy as np
import random

class GridWorld:
    def __init__(self, size=5):
        self.size=size
        self.start=(0,0); self.goal=(size-1,size-1)
        self.state=self.start
        self.actions=[0,1,2,3] # up, right, down, left

    def reset(self):
        self.state=self.start
        return self.state

    def step(self, action):
        r,c=self.state
        if action==0: r=max(0,r-1)
        elif action==1: c=min(self.size-1,c+1)
        elif action==2: r=min(self.size-1,r+1)
        elif action==3: c=max(0,c-1)
        self.state=(r,c)
        done=self.state==self.goal
        reward=10 if done else -1
        return self.state,reward,done

env=GridWorld()
print("Start:",env.reset())
for _ in range(10):
    s,r,d=env.step(random.choice(env.actions))
    print("state=",s,"reward=",r,"done=",d)
    if d: break
