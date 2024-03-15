from logging import Logger

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.trainers.trainer import Trainer
from src.hooks.logger_hook_with_accelerator import LoggerHookWithAccelerator


class AcceleratedTrainer(Trainer):
    def __init__(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        logger: Logger,
        accelerator: Accelerator,
    ) -> None:
        super().__init__(model, data_loader, optimizer)
        self.accelerator = accelerator
        self.model, self.data_loader, self.optimizer = self.accelerator.prepare(
            self.model, self.data_loader, self.optimizer
        )
        self.hooks = [LoggerHookWithAccelerator(logger)]

    def training_step(self) -> torch.Tensor:
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                loss = self.compute_loss()
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            return loss
