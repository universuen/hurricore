from __future__ import annotations

import torch
import transformers
from transformers import PreTrainedModel

from hurricore.trainers import StandardTrainer
from hurricore.hooks import HuggingfaceModelSaveHook
from hurricore.utils import env


class LLMTrainer(StandardTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        # New in this class
        gradient_checkpointing_enable: bool = True,
        # Task specific hyper parameters
        learning_rate: float = 1e-5,
        **kwargs
    ) -> None:
        if gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            **kwargs
        )
        env.logger.info(f"`transformers` version: {transformers.__version__}")
    
    
    def setup_hooks(self) -> None:
        super().setup_hooks()
        self.init_context.hooks.append(
            HuggingfaceModelSaveHook(
                trainer=self,
                folder_path=self.init_context.data_folder_path / "results" / self.init_context.experiment_name,
                interval=self.init_context.ckpt_interval * self.init_context.accelerator.gradient_accumulation_steps,
            )
        )


    def compute_loss(self, batch: dict) -> torch.Tensor:
        model_inputs = batch
        loss = self.model(**model_inputs, use_cache=False).loss
        return loss
