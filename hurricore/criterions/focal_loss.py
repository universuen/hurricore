from typing import Literal

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
            self, 
            alphas: torch.Tensor = None, 
            gamma: float = 2.0, 
            reduction: Literal['none', 'mean', 'sum'] = 'mean',
        ):
        """
        :param alphas: Tensor of shape (C,) where C is the number of classes.
                      This is used to balance the importance of each class.
        :param gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        :param reduction: Specifies the reduction to apply to the output:
                          'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alphas = alphas
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')


    def forward(self, inputs, targets):
        log_probs = self.ce_loss(inputs, targets).float()
        p_t = torch.exp(-log_probs)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * log_probs
        if self.alphas is not None:
            alphas = self.alphas.gather(0, targets)
            focal_loss = focal_loss * alphas
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
