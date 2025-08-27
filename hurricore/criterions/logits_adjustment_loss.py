from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from hurricore.utils import build_label_indices_map


class LogitsAdjustmentLoss(nn.Module):
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
        super().__init__()
        self.class_prob_dist = torch.tensor(class_prob_dist)
        assert self.class_prob_dist.dim() == 1, \
            f"Expected 1-d tensor, got {self.class_prob_dist.dim()}-d tensor."
        assert abs(sum(self.class_prob_dist) - 1) < 1e-3, \
            f"Expected sum of class probability distribution to be 1, got {self.class_prob_dist}."
        self.tau = tau
        self.ce_loss = nn.CrossEntropyLoss()
    

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.class_prob_dist = self.class_prob_dist.to(logits.device)
        logits = logits + self.tau * torch.log(self.class_prob_dist + 1e-12)
        return self.ce_loss(logits, labels)
    

    @staticmethod
    def compute_class_prob_dist(
        dataset: Dataset,
        get_label: callable = lambda x: x[1],
    ):
        label_index = build_label_indices_map(dataset, get_label)
        class_prob_dist = torch.zeros(len(label_index))
        for i in range(len(class_prob_dist)):
            class_prob_dist[i] = len(label_index[i])
        return class_prob_dist / len(dataset)
