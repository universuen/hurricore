import math

import torch
from torch import nn
from torch.nn.utils import spectral_norm


def _init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name == 'Linear':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif 'Norm' in layer_name:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def _sn_conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def _sn_linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class _ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, image_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _sn_conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([hidden_dim, image_size, image_size]),
            nn.LeakyReLU(0.1, inplace=True),
            # _sn_conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


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
        # assert math.log(hidden_dim, 2).is_integer(), 'hidden_dim must be 2^N'
        
        self.hidden_dim = hidden_dim
        self.preprocess_layer = _sn_conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1)
        num_down_samplings = int(math.log(image_size, 2) - 2)
        self.layers = nn.ModuleList()
        for _ in range(num_down_samplings):
            self.layers.extend(
                [
                    _ResBlock(hidden_dim, image_size),
                    _sn_conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.LayerNorm([hidden_dim, image_size // 2, image_size // 2]),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
            )
            image_size //= 2
 
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            _sn_linear(hidden_dim * image_size * image_size, 1, bias=False),
        )
        self.apply(_init_weights)
        
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_layer(z)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
