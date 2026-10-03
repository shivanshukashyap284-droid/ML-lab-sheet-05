# Q28. Study the role of the target network in DQN.
import torch, torch.nn as nn

class Net(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x): return self.net(x)

policy=Net(); target=Net()
target.load_state_dict(policy.state_dict())
x=torch.randn(1,4)
print("Policy Q-values:",policy(x).detach().numpy())
print("Target Q-values:",target(x).detach().numpy())
print("The target network is updated less frequently to stabilize the Q-learning target.")
