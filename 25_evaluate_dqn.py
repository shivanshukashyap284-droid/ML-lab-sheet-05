# Q25. Evaluate a trained DQN agent.
import os, numpy as np, torch, torch.nn as nn, gymnasium as gym

env=gym.make("CartPole-v1"); state_size=4; action_size=2
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2))
    def forward(self,x): return self.net(x)
model=Net()
if not os.path.exists("dqn_cartpole.pth"):
    print("Model not found. Run Q23 first."); raise SystemExit
model.load_state_dict(torch.load("dqn_cartpole.pth",map_location="cpu")); model.eval()
scores=[]
for ep in range(20):
    s,_=env.reset(); total=0; done=False
    while not done:
        with torch.no_grad(): a=int(model(torch.tensor(s,dtype=torch.float32)).argmax())
        s,r,t,tr,_=env.step(a); total+=r; done=t or tr
    scores.append(total)
print("Average evaluation reward:",np.mean(scores))
print("Rewards:",scores)
env.close()
