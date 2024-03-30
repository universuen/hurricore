import torch
from torch import nn

# torch.set_default_device('cuda')

class ResBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, z_dim: 1024) -> None:
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.preprocess_layer = nn.Linear(z_dim, 4 * 4 * 512)
        self.layers = nn.ModuleList(
            [
                ResBlock(512),
                nn.Upsample(scale_factor=2),
            ] * 8
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(-1, self.z_dim)
        x = self.preprocess_layer(z)
        x = x.view(-1, 512, 4, 4)
        for layer in self.layers:
            x = layer(x)
            print(x.shape)
        x = self.final_layer(x)
        return x

model = Generator(1024)
z = torch.randn((64, 1024))
x = model(z)
print(x.shape)
