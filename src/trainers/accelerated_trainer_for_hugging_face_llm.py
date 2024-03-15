from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel

from src.trainers.accelerated_trainer import AcceleratedTrainer


class AcceleratedTrainerForHFLLM(AcceleratedTrainer):
    def __init__(
        self, 
        model: PreTrainedModel, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        logger: Logger,
        accelerator: Accelerator,
    ) -> None:
        super().__init__(model, data_loader, optimizer, logger, accelerator)
    
    def compute_loss(self) -> torch.Tensor:
        input_ids, attention_masks, labels = self.ctx.batch
        loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            use_cache=False,
        )[0]
        return loss
