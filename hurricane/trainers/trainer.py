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
        models: list[nn.Module],
        optimizers: list[Optimizer],
        data_loaders: list[DataLoader],
        accelerator: Accelerator,
        epochs: int = 100,
    ) -> None:
        super().__init__(
            models=models, 
            data_loaders=data_loaders, 
            optimizers=optimizers,
            epochs=epochs,
        )
        # backup original objects
        self.originals = Context()
        self.originals.models = models
        self.originals.data_loaders = data_loaders
        self.originals.optimizers = optimizers
        # setup accelerator
        self.accelerator = accelerator
        self.models = [self.accelerator.prepare(model) for model in models]
        self.optimizers = [self.accelerator.prepare(optimizer) for optimizer in optimizers]
        self.data_loaders = [self.accelerator.prepare(data_loader) for data_loader in data_loaders]

    def training_step(self) -> torch.Tensor:
        for model in self.models:
            model.train()
        with self.accelerator.accumulate(*self.models):
            
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            
            with self.accelerator.autocast():
                loss = self.compute_loss()
            self.accelerator.backward(loss)
            
            for optimizer in self.optimizers:
                optimizer.step()
            
            return loss
