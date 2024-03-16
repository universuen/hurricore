from logging import Logger

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.trainers.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        accelerator: Accelerator,
    ) -> None:
        super().__init__(model, data_loader, optimizer)
        self.accelerator = accelerator
        self.model, self.data_loader, self.optimizer = self.accelerator.prepare(
            self.model, self.data_loader, self.optimizer
        )

    def training_step(self) -> torch.Tensor:
        self.model.train()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                loss = self.compute_loss()
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            return loss
