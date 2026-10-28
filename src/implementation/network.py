import torch
import torch.nn as nn
import torch.nn.functional as F

'''
all we need to do here is implement a basic feedforward network
this network will serve as both our actor and our critic; they however have different output space dims
this is becuase their tasks are different:
  actors output dims is the action space (loss function is eqn 9 in PPO paper)
  critic is just MSE loss (regression task)
    Huber loss may help with early stage noisy training (swiches to MAE for large loss values)
the downstream task for the two is the same, use the network to predict a single value: 
we use the same network and same parameters to predict both the action value (actor) and how well we did (critic) 
'''

class FeedForwardNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, hidden_size)
    self.layer3 = nn.Linear(hidden_size, output_size)
  
  
  def forward(self, x):
    #x is the observation space (our inputs)
    if not isinstance(x, torch.Tensor):
      x = torch.tensor(x, dtype=torch.float32)
    x = self.layer1(x)
    x = F.tanh(x) #tanh preferable to relu in PPO optimization (can experiment with Relu/LeakyReLu)
    x = self.layer2(x)
    x = F.tanh(x)
    out = self.layer3(x)
    
    return out