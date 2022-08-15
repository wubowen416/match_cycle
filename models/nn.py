import torch.nn as nn
import torch.nn.functional as F


class ResLinear(nn.Module):
    def __init__(
        self, 
        d_model: int,
        **kwargs
    ):
        super().__init__()
        self.fc_0 = nn.Linear(d_model, d_model)
        self.fc_1 = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        return x + self.fc_1(F.relu(self.fc_0(x)))