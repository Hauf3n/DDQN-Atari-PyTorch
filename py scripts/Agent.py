import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Network import DQN

class DQN_Agent(nn.Module):
    
    def __init__(self, in_channels, num_actions, epsilon):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = DQN(in_channels, num_actions)
        
        self.eps = epsilon
    
    def forward(self, x):
        actions = self.network(x)
        return actions
    
    def e_greedy(self, x):
        
        actions = self.forward(x)
        
        greedy = torch.rand(1)
        if self.eps < greedy:
            return torch.argmax(actions)
        else:
            return (torch.rand(1) * self.num_actions).type('torch.LongTensor')[0] 
        
    def greedy(self, x):
        actions = self.forward(x)
        return torch.argmax(actions)
    
    def set_epsilon(self, epsilon):
        self.eps = epsilon