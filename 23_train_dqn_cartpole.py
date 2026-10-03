# Q23. Train a DQN agent on CartPole.
import random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
env=gym.make("CartPole-v1")
state_size=env.observation_space.shape[0]; action_size=env.action_space.n

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_size,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,action_size))
    def forward(self,x): return self.net(x)

policy=Net().to(device); target=Net().to(device); target.load_state_dict(policy.state_dict())
optimizer=optim.Adam(policy.parameters(),lr=1e-3)
memory=collections.deque(maxlen=10000)
gamma=.99; batch=64; epsilon=1.0; eps_min=.05; eps_decay=.995
rewards=[]

def learn():
    if len(memory)<batch:return
    b=random.sample(memory,batch)
    s,a,r,ns,d=zip(*b)
    s=torch.tensor(np.array(s),dtype=torch.float32,device=device)
    ns=torch.tensor(np.array(ns),dtype=torch.float32,device=device)
    a=torch.tensor(a,dtype=torch.long,device=device).unsqueeze(1)
    r=torch.tensor(r,dtype=torch.float32,device=device)
    d=torch.tensor(d,dtype=torch.float32,device=device)
    q=policy(s).gather(1,a).squeeze()
    with torch.no_grad(): target_q=r+gamma*(1-d)*target(ns).max(1).values
    loss=nn.functional.mse_loss(q,target_q)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

for ep in range(100):
    s,_=env.reset()
    total=0; done=False
    while not done:
        if random.random()<epsilon:a=env.action_space.sample()
        else:
            with torch.no_grad(): a=int(policy(torch.tensor(s,dtype=torch.float32,device=device)).argmax())
        ns,r,t,tr,_=env.step(a); done=t or tr
        memory.append((s,a,r,ns,done)); s=ns; total+=r
        learn()
    rewards.append(total); epsilon=max(eps_min,epsilon*eps_decay)
    if (ep+1)%10==0: target.load_state_dict(policy.state_dict()); print(ep+1,total)

torch.save(policy.state_dict(),"dqn_cartpole.pth")
env.close()
print("Training finished. Model saved as dqn_cartpole.pth")
