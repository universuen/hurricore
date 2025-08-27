import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int,
        ):
        super().__init__()
        self.proj_1 = nn.Linear(input_dim, output_dim)
        self.proj_2 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x1 = self.proj_1(x)
        x2 = self.proj_2(x)
        return torch.nn.functional.silu(x1) * x2
