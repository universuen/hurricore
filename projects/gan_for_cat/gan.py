import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.utils import spectral_norm

def _init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_dim: int = 1024) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        features_sizes = [512, 256, 128, 64, 32, 16, 8, 3]
        self.layers = nn.ModuleList()
        for in_size, out_size in zip(features_sizes[:-1], features_sizes[1:]):
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_size) if out_size != 3 else nn.Identity(), 
                nn.LeakyReLU(0.2, inplace=True) if out_size != 3 else nn.Sigmoid()  
            )
            self.layers.append(layer)
        
    def forward(self, z: Tensor) -> Tensor:
        assert z.shape[-3:] == (self.z_dim, 1, 1), \
        f"Expected input shape to be (N, {self.z_dim}, 1, 1), got {z.shape}"
        z = self.initial_layer(z)
        for layer in self.layers:
            z = layer(z) 
        return z

    def generate(self, n: int) -> Tensor:
        device = next(self.parameters()).device
        z = torch.randn(n, self.z_dim, 1, 1).to(device)
        return self(z)
    

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features_sizes = [3, 8, 16, 32, 64, 128, 512]
        self.layers = nn.ModuleList()
        for in_size, out_size in zip(features_sizes[:-1], features_sizes[1:]):
            self.layers.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)),
                    nn.GroupNorm(max(out_size // 16, 2), out_size),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == x.shape[-1] == 512, f"Expected input shape to be (B, 512, 512), got {x.shape}"
        for layer in self.layers:
            x = layer(x)
        output = self.final_layer(x)
        return output


class GAN(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()
        self.generator.apply(_init_weights)
        self.discriminator.apply(_init_weights)
