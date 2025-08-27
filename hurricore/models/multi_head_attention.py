import torch
import torch.nn as nn
import math

from hurricore.models.rotary_positional_embedding import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            num_heads: int,
            dropout: float = 0.1,
            use_rope: bool = True,
            max_seq_len: int = 8192,
        ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.head_dim = embedding_dim // num_heads
        
        self.query_projector = nn.Linear(embedding_dim, embedding_dim)
        self.key_projector = nn.Linear(embedding_dim, embedding_dim)
        self.value_projector = nn.Linear(embedding_dim, embedding_dim)
        self.output_projector = nn.Linear(embedding_dim, embedding_dim)

        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
        ) if use_rope else nn.Identity()
        

    def _scaled_dot_product_attention(
            self, 
            query: torch.Tensor,  # (batch_size, num_heads, num_queries, head_dim)
            key: torch.Tensor,  # (batch_size, num_heads, num_keys, head_dim)
            value: torch.Tensor,  # (batch_size, num_heads, num_keys, head_dim)
            mask: torch.Tensor = None,  # (batch_size, num_queries, num_keys)
        ):
        query = self.rope(query)  # (batch_size, num_heads, num_queries, head_dim)
        key = self.rope(key)  # (batch_size, num_heads, num_keys, head_dim)

        attn_scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, num_heads, num_queries, num_keys)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9 if attn_scores.dtype == torch.float32 else -1e+4)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = attn_probs @ value
        return output
    

    def _split_heads(self, x: torch.Tensor):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # (batch_size, num_heads, seq_length, head_dim)
        

    def _combine_heads(self, x: torch.Tensor):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)  # (batch_size, seq_length, embedding_dim)
        

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            mask: torch.Tensor = None,
        ):
        """
        query: torch.Tensor of shape (batch_size, num_queries, embedding_dim)
        key: torch.Tensor of shape (batch_size, num_keys, embedding_dim)
        value: torch.Tensor of shape (batch_size, num_keys, embedding_dim)
        mask: torch.Tensor of shape (batch_size, num_queries, num_keys) or None
        """
        # Mask must be integer tensor with 0s and 1s, where 0s are positions to be masked
        if mask is not None and mask.dtype not in (torch.int, torch.long):
            raise ValueError(f"mask must be integer tensor with 0s and 1s, got {mask.dtype}")
        # Check mask shape
        if mask is not None and mask.shape != (query.shape[0], query.shape[1], key.shape[1]):
            raise ValueError(f"mask shape must be (batch_size, num_queries, num_keys), expected {(query.shape[0], query.shape[1], key.shape[1])}, got {mask.shape}")
        query = self._split_heads(self.query_projector(query))
        key = self._split_heads(self.key_projector(key))
        value = self._split_heads(self.value_projector(value))
        
        attn_output = self._scaled_dot_product_attention(query, key, value, mask)
        output = self.output_projector(self._combine_heads(attn_output))
        return output
