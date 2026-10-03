# Q22. Implement a basic Deep Q-Network using PyTorch.
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(state_size,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,action_size)
        )
    def forward(self,x): return self.net(x)

model=DQN(4,2)
sample=torch.randn(1,4)
print("Network:\n",model)
print("Q-values:",model(sample).detach().numpy())
