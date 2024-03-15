from logging import Logger

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.context import Context


class Trainer:
    def __init__(
        self, 
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: Optimizer,
    ) -> None:

        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.ctx = Context()
        self.hooks = []

    def run(
        self, 
        epochs: int = 1,
    ) -> None:
        self.ctx.epochs = epochs
        
        for hook in self.hooks:
            hook.training_start(self)
        
        for epoch in range(1, epochs + 1):
            self.ctx.epoch = epoch
            
            for hook in self.hooks:
                hook.epoch_start(self)

            for batch_idx, batch in enumerate(self.data_loader):
                self.ctx.batch_idx = batch_idx
                self.ctx.batch = batch

                for hook in self.hooks:
                    hook.iteration_start(self)

                self.ctx.step_loss = self.training_step()

                for hook in self.hooks:
                    hook.iteration_end(self)
            
            for hook in self.hooks:
                hook.epoch_end(self)
        
        for hook in self.hooks:
            hook.training_end(self)

    def training_step(self) -> torch.Tensor:
        self.model.train()
        loss = self.compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def compute_loss(self) -> torch.Tensor:
        raise NotImplementedError
