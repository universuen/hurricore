from __future__ import annotations

from typing import Any

import torch
import transformers
from transformers import PreTrainedModel

from hurricore.trainers import StandardTrainer
from hurricore.utils import env


class LLMTrainer(StandardTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        # Task specific hyper parameters
        num_epochs: int = 3,
        per_device_batch_size: int = 1,
        learning_rate: float = 1e-5,
        gradient_checkpointing_enable: bool = True,
        **kwargs
    ) -> None:
        if gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
        super().__init__(
            model=model,
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            gradient_checkpointing_enable=gradient_checkpointing_enable,
            **kwargs
        )
        env.logger.info(f"`transformers` version: {transformers.__version__}")
    

    def compute_loss(self, batch: dict) -> torch.Tensor:
        model_inputs = batch
        loss = self.model(**model_inputs, use_cache=False).loss
        return loss
