from typing import Literal

import torch
from torch import nn


class BinaryFocalLoss(nn.Module):
    def __init__(
            self, 
            alpha: float = 0.25, 
            gamma: float = 2.0, 
            reduction: Literal['none', 'mean', 'sum'] = 'mean',
    ):
        """
        :param alpha: Weighting factor in the range (0,1) to balance positive vs negative examples.
        :param gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        :param reduction: Specifies the reduction to apply to the output:
                          'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(logits, labels)
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
