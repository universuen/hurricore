import torch
import torch.nn as nn


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) regularization module.

    Args:
        p (float): Probability of dropping a path. Default: 0.0
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p


    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        output = x * binary_mask / keep_prob
        return output


    def extra_repr(self):
        return f'p={self.p}'
