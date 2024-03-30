import math

import torch
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self, 
        z_dim: int = 1024, 
        image_size: int = 512, 
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        
        assert image_size >= 4, 'image_size must be at least 4'
        assert math.log(image_size, 2).is_integer(), 'image_size must be 2^N'
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.preprocess_layer = nn.Linear(z_dim, hidden_dim * 4 * 4)
        num_up_samples = int(math.log(image_size, 2) - 2)
        self.layers = nn.ModuleList(
            [
                _ResBlock(hidden_dim),
                nn.Upsample(scale_factor=2),
            ] * num_up_samples
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_layer(z)
        x = x.view(-1, self.hidden_dim, 4, 4)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
