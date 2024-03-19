from logging import Logger

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hurricane.context import Context


class TrainerBase:
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
        self.ctx.epoch = 0
        
        for hook in self.hooks:
            hook.on_training_start(self)
            
        for epoch in range(self.ctx.epoch + 1, epochs + 1):
            self.ctx.epoch = epoch
            
            for hook in self.hooks:
                hook.on_epoch_start(self)
                
            for batch_idx, batch in enumerate(self.data_loader):
                self.ctx.batch_idx = batch_idx
                self.ctx.batch = batch

                for hook in self.hooks:
                    hook.on_step_start(self)

                self.ctx.step_loss = self.training_step()

                for hook in self.hooks:
                    hook.on_step_end(self)
            
            for hook in self.hooks:
                hook.on_epoch_end(self)
        
        for hook in self.hooks:
            hook.on_training_end(self)

    def training_step(self) -> Tensor:
        self.model.train()
        loss = self.compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def compute_loss(self) -> Tensor:
        raise NotImplementedError
