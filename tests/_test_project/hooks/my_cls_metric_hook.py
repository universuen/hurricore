from __future__ import annotations

from typing import Any
from torch._tensor import Tensor
from hurricore.hooks import ClassificationEvaluationHook



class MyClsEvaluationHook(ClassificationEvaluationHook):

    def get_probs_and_labels(self, batch: Any) -> tuple[Tensor, Tensor]:
        x, y = batch
        pred = self.trainer.model(x)
        # prob = pred.softmax(dim=-1)
        return pred, y

