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
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
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


class Discriminator(nn.Module):
    def __init__(
        self, 
        image_size: int = 256, 
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        assert image_size >= 4, 'image_size must be at least 4'
        assert hidden_dim >= image_size, 'hidden_dim must be at least image_size'
        assert math.log(image_size, 2).is_integer(), 'image_size must be 2^N'
        assert math.log(hidden_dim, 2).is_integer(), 'hidden_dim must be 2^N'
        
        self.hidden_dim = hidden_dim
        self.preprocess_layer = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        num_down_samplings = int(math.log(image_size, 2) - 2)
        self.layers = nn.ModuleList()
        for _ in range(num_down_samplings):
            self.layers.extend(
                [
                    _ResBlock(hidden_dim),
                    nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim // 2),
                ]
            )
            hidden_dim //= 2
 
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 4, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.apply(init_weights)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_layer(z)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
