# Q14. Compare exploration/exploitation using different epsilon values.
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def train(epsilon, episodes=2000):
    env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4)); rewards=[]
    for _ in range(episodes):
        s,_=env.reset(); total=0; done=False
        while not done:
            a=np.argmax(q[s]) if np.random.rand()>epsilon else env.action_space.sample()
            ns,r,term,trunc,_=env.step(a)
            q[s,a]+=.8*(r+.95*np.max(q[ns])-q[s,a])
            s,total,done=ns,total+r,term or trunc
        rewards.append(total)
    env.close(); return np.mean(rewards)
for e in [0.0,0.1,0.5,1.0]:
    print(f"Epsilon={e}: average training reward={train(e):.3f}")
