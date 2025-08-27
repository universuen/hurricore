import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 4096, 
        base: float = 10000.0,
    ):
        """
        Rotary Positional Embedding (RoPE) implementation without complex numbers.
        
        Args:
            dim (int): Dimension of the embedding (should be even)
            max_seq_len (int): Maximum sequence length to support
            base (float): Base for frequency computation
        """
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        self.freqs: torch.Tensor
        self.compute_freqs()
    

    def compute_freqs(self) -> None:
        """
        Precompute sine and cosine frequencies.
        
        Returns:
            torch.Tensor: Precomputed sine and cosine frequencies
        """
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len).float()
        sinusoidal_inp = torch.outer(t, inv_freq)
        cos_emb = sinusoidal_inp.cos()
        sin_emb = sinusoidal_inp.sin()
        self.freqs = cos_emb, sin_emb
    

    def _rotate(
            self, 
            x: torch.Tensor, 
            cos_emb: torch.Tensor, 
            sin_emb: torch.Tensor,
        ):
        """
        Rotate the input tensor using precomputed sine and cosine embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, dim)
            cos_emb (torch.Tensor): Cosine embedding
            sin_emb (torch.Tensor): Sine embedding
        
        Returns:
            torch.Tensor: Rotated tensor
        """
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even and odd indices
        rotated_x1 = x1 * cos_emb - x2 * sin_emb
        rotated_x2 = x1 * sin_emb + x2 * cos_emb
        return torch.stack((rotated_x1, rotated_x2), dim=-1).reshape(x.shape)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to queries or keys.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, dim)
        
        Returns:
            torch.Tensor: Rotated tensor
        """
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum supported length {self.max_seq_len}")
        cos_emb, sin_emb = self.freqs
        cos_emb = cos_emb[:seq_len].to(x.device).unsqueeze(0)  # Adjust shape
        sin_emb = sin_emb[:seq_len].to(x.device).unsqueeze(0)
        return self._rotate(x, cos_emb, sin_emb)
