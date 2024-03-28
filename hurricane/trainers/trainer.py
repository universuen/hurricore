from logging import Logger

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers import TrainerBase
from hurricane.context import Context


class Trainer(TrainerBase):
    def __init__(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        accelerator: Accelerator,
        epochs: int = 100,
    ) -> None:
        super().__init__(
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer,
            epochs=epochs,
        )
        # backup original objects
        self.originals = Context()
        self.originals.model = model
        self.originals.data_loader = data_loader
        self.originals.optimizer = optimizer
        # setup accelerator
        self.accelerator = accelerator
        self.model, self.data_loader, self.optimizer = self.accelerator.prepare(
            self.model, self.data_loader, self.optimizer
        )

    def training_step(self) -> torch.Tensor:
        self.model.train()
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with self.accelerator.autocast():
                loss = self.compute_loss()
            self.accelerator.backward(loss)
            self.optimizer.step()
            return loss
