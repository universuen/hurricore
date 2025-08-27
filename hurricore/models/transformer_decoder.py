import torch
from torch import nn

from hurricore.models.transformer_decoder_layer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):
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
                TransformerDecoderLayer(embedding_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        
        
    def forward(
            self, 
            x: torch.Tensor, 
            memory: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            cross_attn_mask: torch.Tensor = None,
        ):
        """
        x: shape (batch_size, query_seq_length, embedding_dim)
        memory: shape (batch_size, memory_seq_length, embedding_dim)
        self_attn_mask: shape (batch_size, query_seq_length, query_seq_length) or None
        cross_attn_mask: shape (batch_size, query_seq_length, memory_seq_length) or None
        
        returns: shape (batch_size, query_seq_length, embedding_dim)
        """
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask, cross_attn_mask)
        return x
