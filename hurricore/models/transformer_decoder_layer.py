import torch
from torch import nn

from hurricore.models.multi_head_attention import MultiHeadAttention
from hurricore.models.swi_glu import SwiGLU
from hurricore.models.rms_norm import RMSNorm
from hurricore.models.moe import MoE


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            num_heads: int, 
            ffn_dim: int = None, 
            dropout: float = 0.1,
            # MoE related config
            use_moe: bool = False,
            moe_expert_dim: int = None,
            moe_num_shared_experts: int = 2,
            moe_num_routed_experts: int = 64,
            moe_top_k: int = 6,
        ):
        super().__init__()

        ffn_dim = ffn_dim or 4 * embedding_dim
        moe_expert_dim = moe_expert_dim or ffn_dim // (moe_num_shared_experts + moe_num_routed_experts)
        
        self.self_attn = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attn = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = MoE(
            input_dim=embedding_dim,
            expert_dim=moe_expert_dim,
            num_shared_experts=moe_num_shared_experts,
            num_routed_experts=moe_num_routed_experts,
            top_k=moe_top_k,
        ) if use_moe else nn.Sequential(
            SwiGLU(embedding_dim, ffn_dim),
            nn.Linear(ffn_dim, embedding_dim),
        )
        self.norm_1 = RMSNorm(embedding_dim)
        self.norm_2 = RMSNorm(embedding_dim)
        self.norm_3 = RMSNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        

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
        # self-attention
        x_normed = self.norm_1(x)
        attn_output = self.self_attn(x_normed, x_normed, x_normed, self_attn_mask)
        x = x + self.dropout(attn_output)
        
        # cross-attention
        x_normed = self.norm_2(x)
        cross_attn_output = self.cross_attn(x_normed, memory, memory, cross_attn_mask)
        x = x + self.dropout(cross_attn_output)
        
        # feed-forward
        x_normed = self.norm_3(x)
        ff_output = self.ffn(x_normed)
        x = x + self.dropout(ff_output)
        
        return x
