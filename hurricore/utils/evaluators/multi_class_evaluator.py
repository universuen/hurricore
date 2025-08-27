from __future__ import annotations
from typing import Any, Literal

import torch
from torch import nn

from hurricore.utils.evaluators.classification_evaluator import ClassificationEvaluator


class MultiClassEvaluator(ClassificationEvaluator):
    def __init__(
            self, 
            num_classes: int,
            threshold: float = 0.5,
            average: Literal["micro", "macro", "weighted", "none"] = "none",
        ):
        super().__init__(
            task="multiclass",
            num_classes=num_classes,
            average=average,
            threshold=threshold,
        )


    def get_probs_and_labels(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        model_inputs, labels = batch
        assert type(model_inputs) is dict
        logits = model(**model_inputs)  # [batch_size, num_labels]
        probs = torch.softmax(logits, dim=-1)  # [batch_size, num_labels]
        return probs, labels
