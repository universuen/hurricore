import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
