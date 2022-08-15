import torch.nn as nn
import torch.nn.functional as F

from .nn import ResLinear


class ResLinearNet(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        d_model : int, 
        num_layers: int,
        **kwargs
    ):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU())
        self.res_blocks = nn.ModuleList([ResLinear(d_model) for _ in range(num_layers)])
        self.output_layer = nn.Sequential(nn.Linear(d_model, d_out))
        if "dropout" in kwargs.keys():
            self.dropout = nn.Dropout(kwargs['dropout'])

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.res_blocks:
            if hasattr(self, "dropout"):
                h = self.dropout(block(h))
            else:
                h = block(h)
        return self.output_layer(h)