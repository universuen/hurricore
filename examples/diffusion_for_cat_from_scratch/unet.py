import math

import torch
from torch import nn


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


class SinusoidalPositionalEmbedding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, _, embed_dim = x.shape
        pe = torch.zeros(seq_len, embed_dim, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).to(x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(1)


class ViT(nn.Module):
    def __init__(
        self,
        num_channels: int = 3, 
        embed_dim: int = 256, 
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        flatten_dim = patch_size * patch_size * num_channels
        self.projector_in = nn.Linear(flatten_dim, embed_dim)
        self.positional_embedding = SinusoidalPositionalEmbedding()
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.projector_out = nn.Linear(embed_dim, flatten_dim)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, image_size, _ = x.shape
        num_patches = (image_size // self.patch_size) ** 2
        x = patchify(x, self.patch_size)
        x = x.view(batch_size, num_patches, -1)
        x = self.projector_in(x)
        x = self.positional_embedding(x)
        x = self.attention_layer(x, x, x)[0]
        x = self.projector_out(x)
        x = x.view(batch_size, num_patches, num_channels, self.patch_size, self.patch_size)
        x = unpatchify(x, image_size=image_size)
        return x


class DownSampling(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        attn_embed_dim: int = 256,
        attn_patch_size: int = 32,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, in_dim) if in_dim != 3 else nn.Identity(),
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.1),
        )
        self.attn = nn.Sequential(
            nn.GroupNorm(8, out_dim),
            ViT(out_dim, attn_embed_dim, attn_patch_size),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conved_x = self.conv(x)
        attened_x = self.attn(conved_x)
        return conved_x + attened_x


class UpSampling(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        attn_embed_dim: int = 256,
        attn_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, in_dim),
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.1),
        )
        self.attn = nn.Sequential(
            nn.GroupNorm(8, out_dim) if out_dim != 3 else nn.Identity(),
            ViT(out_dim, attn_embed_dim, attn_patch_size),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conved_x = self.conv(x)
        attened_x = self.attn(conved_x)
        return conved_x + attened_x


def sinusoidal_embedding(n, d):
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)
    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])
    return embedding


class UNet(nn.Module):
    def __init__(
        self, 
        image_size=256,
        hidden_dim=16, 
        attn_embed_dim=64,
        attn_patch_size=32,
        num_steps=1000,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_steps = num_steps
        
        self.time_embedding = nn.Embedding(num_steps, image_size ** 2 * 3)
        self.time_embedding.weight.data = sinusoidal_embedding(num_steps, image_size ** 2 * 3)
        self.time_embedding.requires_grad_(False)

        self.down1 = DownSampling(3, hidden_dim, attn_embed_dim, attn_patch_size)
        self.down2 = DownSampling(hidden_dim, hidden_dim * 2, attn_embed_dim, attn_patch_size)
        self.down3 = DownSampling(hidden_dim * 2, hidden_dim * 4, attn_embed_dim, attn_patch_size)
        self.down4 = DownSampling(hidden_dim * 4, hidden_dim * 8, attn_embed_dim, attn_patch_size)

        
        self.bottleneck = nn.Sequential(
            nn.GroupNorm(8, hidden_dim * 8),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.GroupNorm(8, hidden_dim * 16),
            ViT(hidden_dim * 16, attn_embed_dim, attn_patch_size),
            nn.GroupNorm(8, hidden_dim * 16),
            nn.Conv2d(hidden_dim * 16, hidden_dim * 8, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        self.up1 = UpSampling(hidden_dim * 8 * 2, hidden_dim * 4, attn_embed_dim, attn_patch_size)
        self.up2 = UpSampling(hidden_dim * 4 * 2, hidden_dim * 2, attn_embed_dim, attn_patch_size)
        self.up3 = UpSampling(hidden_dim * 2 * 2, hidden_dim, attn_embed_dim, attn_patch_size)
        self.up4 = UpSampling(hidden_dim * 2, 3, attn_embed_dim, attn_patch_size)
        
        self.final = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(t).view(-1, 3, self.image_size, self.image_size)
        x = x + time_embedding
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        b = self.bottleneck(d4)
        
        u1 = self.up1(torch.cat([b, d4], dim=1))
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        
        out = self.final(u4)
        return out
