# Q13. Study the effect of different Gamma values.
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def run(gamma, episodes=2000):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    q=np.zeros((16,4)); rewards=[]
    for _ in range(episodes):
        s,_=env.reset(); total=0; done=False
        while not done:
            a=np.argmax(q[s]) if np.random.rand()>.1 else env.action_space.sample()
            ns,r,term,trunc,_=env.step(a)
            q[s,a]+=.8*(r+gamma*np.max(q[ns])-q[s,a])
            s,total,done=ns,total+r,term or trunc
        rewards.append(total)
    env.close()
    return np.convolve(rewards,np.ones(100)/100,mode="valid")

for gamma in [0.5,0.8,0.99]:
    plt.plot(run(gamma),label=f"gamma={gamma}")
plt.xlabel("Episode"); plt.ylabel("Moving Average Reward")
plt.title("Effect of Discount Factor")
plt.legend(); plt.show()
