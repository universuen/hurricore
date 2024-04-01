from typing import Iterable
from itertools import zip_longest

from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hurricane.context import Context


class TrainerBase:
    def __init__(
        self, 
        models: list[nn.Module],
        optimizers: list[Optimizer],
        data_loaders: list[DataLoader],
        epochs: int = 100,
    ) -> None:

        self.models = models
        self.optimizers = optimizers
        self.data_loaders = data_loaders
        self.ctx = Context()
        self.epochs = epochs
        self.hooks = []
        
        assert len(set([len(dl) for dl in data_loaders])) == 1, 'All data loaders must have the same length.'
    
    
    def run(self) -> None:

        self.ctx.epoch = 0
        self.ctx.batches_idx = 0
        
        for hook in self.hooks:
            hook.on_training_start()
            
        for epoch in range(self.ctx.epoch, self.epochs):
            self.ctx.epoch = epoch
            
            for hook in self.hooks:
                hook.on_epoch_start()
            
            for batches_idx, batches in enumerate(
                iterable=self.build_iterator(), 
                start=self.ctx.batches_idx
            ):
                self.ctx.batches_idx = batches_idx
                self.ctx.batches = batches
                self._set_global_step()
                
                for hook in self.hooks:
                    hook.on_step_start()

                self.ctx.step_loss = self.training_step()

                for hook in self.hooks:
                    hook.on_step_end()
            
            for hook in self.hooks:
                hook.on_epoch_end()
                
            self.ctx.batches_idx = 0
        
        for hook in self.hooks:
            hook.on_training_end()
    
    
    def build_iterator(self) -> Iterable:
        self.ctx.iterator_length = max([len(dl) for dl in self.data_loaders])
        return zip_longest(*self.data_loaders, fillvalue=None)
    
    
    def training_step(self) -> Tensor:
        
        for model in self.models:
            model.train()
            
        for optimizer in self.optimizers:
            optimizer.zero_grad()
  
        loss = self.compute_loss()
        loss.backward()
        
        for optimizer in self.optimizers:
            optimizer.step()
        
        return loss
    
    
    def compute_loss(self) -> Tensor:
        raise NotImplementedError

    
    def get_hook(self, hook_type):
        for hook in self.hooks:
            if isinstance(hook, hook_type):
                return hook
        return None

    
    def _set_global_step(self) -> int:
        epoch = self.ctx.epoch
        iterator_length = self.ctx.iterator_length
        batches_idx = self.ctx.batches_idx
        self.ctx.global_step = epoch * iterator_length + batches_idx

    
    def __repr__(self) -> str:
        result = f'{self.__class__.__name__}(\n'
        for hook in self.hooks:
            result += f'    {hook.__class__.__name__},\n'
        result += ')'
        return result
    