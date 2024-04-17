import math

import torch
from torch import nn


def embed_to_sinusoidal(time_steps: torch.Tensor, dim: int) -> torch.Tensor:
    assert len(time_steps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=time_steps.device
    )
    exponent = exponent / (half_dim - 1)
    emb = torch.exp(exponent)
    emb = time_steps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = embed_to_sinusoidal(t, self.dim)
        return self.layers(emb)


def patchify(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    assert x.dim() == 4, "Input tensor should be [B, C, H, W]"
    batch_size, num_channels, h, w = x.shape
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.contiguous().view(batch_size, num_channels, -1, patch_size, patch_size)
    return x.permute(0, 2, 1, 3, 4).contiguous()


def unpatchify(x: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    assert x.dim() == 5, "Input tensor should be [B, N, C, H, W]"
    batch_size, num_patches, num_channels, patch_size, _ = x.shape
    num_patches_per_line = image_size // patch_size
    assert num_patches_per_line ** 2 == num_patches, "num_patches should be a square number"

    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(batch_size, num_channels, num_patches_per_line, num_patches_per_line, patch_size, patch_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    imgs = x.view(batch_size, num_channels, image_size, image_size)
    return imgs


class ViT(nn.Module):
    def __init__(
        self,
        num_channels: int = 3, 
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = num_channels * patch_size ** 2
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=(patch_size * 4) ** 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

    def bk_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, image_size, _ = x.shape
        num_patches = (image_size // self.patch_size) ** 2
        
        x = patchify(x, self.patch_size)
        x = x.view(batch_size, num_patches, -1)
        
        x = x + embed_to_sinusoidal(torch.arange(num_patches, device=x.device), self.embed_dim)[None, :, :]
        x = self.attention_layer(x, x, x)[0]
        
        x = x.view(batch_size, num_patches, num_channels, self.patch_size, self.patch_size)
        x = unpatchify(x, image_size=image_size)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, image_size, _ = x.shape
        x = x.view(batch_size, num_channels, -1)
        
        x = x + embed_to_sinusoidal(torch.arange(num_channels, device=x.device), image_size ** 2)[None, :, :]
        x = self.attention_layer(x, x, x)[0]
        
        x = x.view(batch_size, num_channels, image_size, image_size)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        with_attention: bool = True,
        attention_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.with_attention = with_attention
        
        self.conv_block_1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_embedding_projection = nn.Linear(time_embedding_dim, out_channels)
        self.conv_block_2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if self.with_attention:
            self.attention_block = nn.Sequential(
                nn.GroupNorm(32, out_channels),
                nn.SiLU(),
                ViT(
                    num_channels=out_channels,
                    patch_size=attention_patch_size,
                )
            )
        
        self.x_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.conv_block_1(x)
        time_emb = self.time_embedding_projection(t)
        hidden_states = hidden_states + time_emb[:, :, None, None]
        hidden_states = self.conv_block_2(hidden_states)
        
        if self.with_attention:
            hidden_states = self.attention_block(hidden_states)
            
        projected_x = self.x_projection(x)
        return hidden_states + projected_x
        
        

class DownSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_residual_blocks: int,
        with_attention: bool = True,
        attention_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.num_residual_blocks = num_residual_blocks
        self.with_attention = with_attention
        self.attention_patch_size = attention_patch_size
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    time_embedding_dim=time_embedding_dim,
                    with_attention=with_attention,
                    attention_patch_size=attention_patch_size,
                )
            )
            in_channels = out_channels
        self.down_sampling = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for block in self.residual_blocks:
            x = block(x, t)
        return self.down_sampling(x)
        

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_residual_blocks: int,
        with_attention: bool = True,
        attention_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.num_residual_blocks = num_residual_blocks
        self.with_attention = with_attention
        self.attention_patch_size = attention_patch_size
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    time_embedding_dim=time_embedding_dim,
                    with_attention=with_attention,
                    attention_patch_size=attention_patch_size,
                )
            )
            in_channels = out_channels
        
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for block in self.residual_blocks:
            x = block(x, t)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_residual_blocks: int,
        with_attention: bool = True,
        attention_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.num_residual_blocks = num_residual_blocks
        self.with_attention = with_attention
        self.attention_patch_size = attention_patch_size
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    time_embedding_dim=time_embedding_dim,
                    with_attention=with_attention,
                    attention_patch_size=attention_patch_size,
                )
            )
            in_channels = out_channels
        self.up_sampling = nn.Upsample(scale_factor=2)
    
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for block in self.residual_blocks:
            x = block(x, t)
        return self.up_sampling(x)


class UNet(nn.Module):
    def __init__(
        self, 
        image_size=256,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256),
        time_embedding_dim=512,
    ): 
        super().__init__()
        self.image_size = image_size
        self.layers_per_block = layers_per_block
        self.block_out_channels = block_out_channels
        self.time_embedding = TimeEmbedding(dim=time_embedding_dim)
        
        self.pre_conv = nn.Conv2d(3, block_out_channels[0], kernel_size=3, padding=1)
        channels = (block_out_channels[0], ) + block_out_channels

        current_image_size = image_size
        
        self.down_blocks = nn.ModuleList()
        for idx, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.down_blocks.append(
                DownSamplingBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                    num_residual_blocks=layers_per_block,
                    with_attention=True if idx == len(channels) - 3 else False,
                    attention_patch_size=current_image_size // 4,
                )
            )
            current_image_size //= 2
            
        self.bottleneck_block = BottleneckBlock(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            time_embedding_dim=time_embedding_dim,
            num_residual_blocks=layers_per_block,
            with_attention=False,
        )
        
        self.up_blocks = nn.ModuleList()
        for idx, (in_channels, out_channels) in enumerate(zip(channels[-1:0:-1], channels[-2::-1])):
            in_channels *= 2
            self.up_blocks.append(
                UpSamplingBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                    num_residual_blocks=layers_per_block,
                    with_attention=True if idx == 1 else False,
                    attention_patch_size=current_image_size // 4,
                )
            )
            current_image_size *= 2
            
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, block_out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        x = self.pre_conv(x)
        t = self.time_embedding(t)
        
        residual_states = []
        for block in self.down_blocks:
            x = block(x, t)
            residual_states.append(x)
        
        x = self.bottleneck_block(x, t)
        
        for x_prev, block in zip(reversed(residual_states), self.up_blocks):
            x = torch.cat([x, x_prev], dim=1)
            x = block(x, t)
            
        return self.final_conv(x)

