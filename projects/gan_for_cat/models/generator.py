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


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name == 'Linear':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif 'Norm' in layer_name:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


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
                nn.Upsample(scale_factor=4),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            ] * num_up_samples
        )
        self.final_layer = nn.Sequential(
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            _ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.apply(init_weights)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_layer(z)
        x = x.view(-1, self.hidden_dim, 4, 4)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
