from __future__ import annotations

import torch

from hurricore.criterions.logits_adjustment_loss import LogitsAdjustmentLoss


class BinaryLogitsAdjustmentLoss(LogitsAdjustmentLoss):
    def __init__(
            self, 
            class_prob_dist: torch.Tensor | list[float],
            tau: float = 1.0,
        ) -> None:
        """
        :param class_prob_dist: Tensor of shape (C,) where C is the number of classes.
                                This is used to adjust the logits of the model.
        :param tau: Parameter that scales the logits adjustment.
        """
        super().__init__(
            class_prob_dist=class_prob_dist,
            tau=tau,
        )
        self.ce_loss = torch.nn.BCEWithLogitsLoss()
    
