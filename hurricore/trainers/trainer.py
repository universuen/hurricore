from typing import Iterable
from itertools import zip_longest

from accelerate import Accelerator
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hurricore.utils import Context


class Trainer:
    def __init__(
        self, 
        models: list[nn.Module],
        optimizers: list[Optimizer],
        data_loaders: list[DataLoader],
        accelerator: Accelerator,
        num_epochs: int = 100,
    ) -> None:
        # backup original objects
        self.originals = Context(
            models=models,
            data_loaders=data_loaders,
            optimizers=optimizers,
        )
        # setup accelerated objects (ugly but necessary when using DeepSpeed)
        all_accelerated_objects = accelerator.prepare(*models, *data_loaders, *optimizers)
        self.models = all_accelerated_objects[:len(models)]
        self.data_loaders = all_accelerated_objects[len(models):len(models) + len(data_loaders)]
        self.optimizers = all_accelerated_objects[len(models) + len(data_loaders):]
        self.accelerator = accelerator
        # setup context and number of epochs
        self.ctx = Context(num_epochs=num_epochs)
        # initialize hooks list
        self.hooks = []
    
    
    def run(self) -> None:
        # initialize context variables
        self.ctx.epoch = 0
        self.ctx.batches_idx = 0
        # execute hooks on training start
        for hook in self.hooks:
            hook.on_training_start()
        # iterate over epochs
        for epoch in range(self.ctx.epoch, self.ctx.num_epochs):
            # update context variables
            self.ctx.epoch = epoch
            # execute hooks on epoch start
            for hook in self.hooks:
                hook.on_epoch_start()
            # iterate over batches
            for batches_idx, batches in enumerate(
                iterable=self.build_iterator(), 
                start=self.data_loaders[0].skip_batches
            ):
                # update context variables
                self.ctx.batches_idx = batches_idx
                self.ctx.batches = batches
                self._set_global_step()
                # execute hooks on step start
                for hook in self.hooks:
                    hook.on_step_start()
                # execute training step and collect loss
                self.ctx.step_loss = self.training_step()
                # execute hooks on step end
                for hook in self.hooks:
                    hook.on_step_end()
            # execute hooks on epoch end
            for hook in self.hooks:
                hook.on_epoch_end()
        # execute hooks on training end
        for hook in self.hooks:
            hook.on_training_end()
    
    
    def build_iterator(self) -> Iterable:
        # build iterator for data loaders and add number of steps per epoch to context
        self.ctx.num_steps_per_epoch = max([len(dl) for dl in self.data_loaders])
        return zip_longest(*self.data_loaders, fillvalue=None)

    
    def training_step(self) -> Tensor:
        # set models to training mode
        for model in self.models:
            model.train()
        # wrap forward and backward passes to enable gradient accumulation
        with self.accelerator.accumulate(*self.models):
            # zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            # forward pass with mixed precision
            with self.accelerator.autocast():
                loss = self.compute_loss()
            # backward pass to compute gradients
            self.accelerator.backward(loss)
            # update parameters using gradients
            for optimizer in self.optimizers:
                optimizer.step()
            # return loss for context tracking
            return loss
    
    
    def compute_loss(self) -> Tensor:
        raise NotImplementedError
    
    
    def get_hook(self, hook_type):
        # get hook of specific type
        for hook in self.hooks:
            if isinstance(hook, hook_type):
                return hook
        return None


    def _set_global_step(self) -> int:
        epoch = self.ctx.epoch
        num_steps_per_epoch = self.ctx.num_steps_per_epoch
        batches_idx = self.ctx.batches_idx
        self.ctx.global_step = epoch * num_steps_per_epoch + batches_idx
    
    
    def __repr__(self) -> str:
        result = f'{self.__class__.__name__}(\n'
        for hook in self.hooks:
            result += f'    {hook.__class__.__name__},\n'
        result += ')'
        return result
    