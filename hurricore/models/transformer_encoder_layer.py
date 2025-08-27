import torch
from torch import nn

from hurricore.models.multi_head_attention import MultiHeadAttention
from hurricore.models.swi_glu import SwiGLU
from hurricore.models.rms_norm import RMSNorm
from hurricore.models.moe import MoE


class TransformerEncoderLayer(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(
            self, 
            x: torch.Tensor, 
            mask:torch.Tensor = None,
        ):
        """
        x: shape (batch_size, seq_length, embedding_dim)
        mask: shape (batch_size, seq_length) or None
        
        returns: shape (batch_size, seq_length, embedding_dim)
        """

        x_normed = self.norm_1(x)
        attn_output = self.self_attn(x_normed, x_normed, x_normed, mask)
        x = x + self.dropout(attn_output)

        x_normed = self.norm_2(x)
        ff_output = self.ffn(x_normed)
        x = x + self.dropout(ff_output)
        
        return x
