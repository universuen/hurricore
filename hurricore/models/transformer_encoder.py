import torch
from torch import nn

from hurricore.models.transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            num_layers: int = 6, 
            embedding_dim: int = 512, 
            num_heads: int = 8, 
            ff_dim: int = 2048, 
            dropout: float = 0.1,
        ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        
        
    def forward(
            self, 
            x: torch.Tensor, 
            mask:torch.Tensor = None,
        ):
        """
        x: shape (batch_size, seq_length, embedding_dim)
        mask: shape (batch_size, seq_length, seq_length) None
        
        returns: shape (batch_size, seq_length, embedding_dim)
        """
        mask = mask.int() if mask is not None else None
        for layer in self.layers:
            x = layer(x, mask)
        return x
