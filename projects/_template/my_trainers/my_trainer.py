from typing import Any

import torch

from hurricore.trainers import StandardTrainer


class MyTrainer(StandardTrainer):
    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError
