from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Any
from itertools import zip_longest

from accelerate import Accelerator
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hurricore.utils import AttributeDict, HurricoreHookList

if TYPE_CHECKING:
    from hurricore.hooks import HurricoreHook


class _Originals(AttributeDict):
    models: list[nn.Module]
    optimizers: list[Optimizer]
    data_loaders: list[DataLoader]


class _Context(AttributeDict):
    num_epochs: int
    epoch: int
    batch_idx: int
    batch: tuple[Tensor] | Tensor
    step_loss: Tensor
    num_steps_per_epoch: int

    @property
    def global_step(self) -> int:
        epoch = self.epoch
        num_steps_per_epoch = self.num_steps_per_epoch
        batch_idx = self.batch_idx
        return epoch * num_steps_per_epoch + batch_idx

    @property
    def total_steps(self) -> int:
        num_epochs = self.num_epochs
        num_steps_per_epoch = self.num_steps_per_epoch
        return num_epochs * num_steps_per_epoch


class HurricoreTrainer:
    def __init__(
        self,
        models: list[nn.Module],
        optimizers: list[Optimizer],
        data_loaders: list[DataLoader],
        accelerator: Accelerator,
        num_epochs: int,
        gradient_clipping_value: float | None = None,
        gradient_clipping_norm: float | None = None,
        hooks: list[HurricoreHook] | None = None,
    ) -> None:
        # Backup original objects
        self.originals = _Originals(
            models=models,
            data_loaders=data_loaders,
            optimizers=optimizers,
        )
        # Setup accelerated objects (ugly but necessary when using DeepSpeed)
        all_accelerated_objects = accelerator.prepare(
            *models, *optimizers, *data_loaders
        )
        num_models = len(models)
        num_optimizers = len(optimizers)
        self.models: list[nn.Module] = all_accelerated_objects[:num_models]
        self.optimizers: list[Optimizer] = all_accelerated_objects[num_models:num_models + num_optimizers]
        self.data_loaders: list[DataLoader] = all_accelerated_objects[num_models + num_optimizers:]
        self.accelerator: Accelerator = accelerator
        # Set easy access to main objects
        self.model, self.original_model = self.models[0], self.originals.models[0]
        self.optimizer, self.original_optimizer = self.optimizers[0], self.originals.optimizers[0]
        self.data_loader, self.original_data_loader = self.data_loaders[0], self.originals.data_loaders[0]
        # Setup context and number of epochs
        num_steps_per_epoch = max(len(dl) for dl in self.data_loaders)
        self.context = _Context(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
        )
        # Set gradient clipping
        self.gradient_clipping_value = gradient_clipping_value
        self.gradient_clipping_norm = gradient_clipping_norm
        # Setup hooks
        self.hooks: HurricoreHookList[HurricoreHook] = HurricoreHookList(hooks or [])
        [hook.execute_on_trainer_init_end() for hook in self.hooks]


    @staticmethod
    def _build_data_iterator(data_loaders: list[DataLoader]) -> Iterable:
        if len(data_loaders) == 1:
            return data_loaders[0]
        else:
            return zip_longest(*data_loaders, fillvalue=None)


    def run(self) -> None:
        # Initialize context variables
        self.context.epoch = 0
        self.context.batch_idx = 0
        self.context.global_step = -1
        # execute hooks on training start
        [hook.execute_on_training_start() for hook in self.hooks]
        
        # Iterate over epochs
        for epoch in range(self.context.epoch, self.context.num_epochs):
            # Update context variables
            self.context.epoch = epoch
            # Execute hooks on epoch start
            [hook.execute_on_epoch_start() for hook in self.hooks]
            # Iterate over batches
            for batch_idx, batch in enumerate(
                iterable=self._build_data_iterator(self.data_loaders),
                start=self.context.batch_idx,
            ):
                # Update context variables
                self.context.batch_idx = batch_idx
                self.context.batch = batch
                # Execute hooks on step start
                [hook.execute_on_step_start() for hook in self.hooks]
                # Step on batch and collect loss
                self.context.step_loss = self.step(batch)
                # Execute hooks on step end
                [hook.execute_on_step_end() for hook in self.hooks]
            # Execute hooks on epoch end
            [hook.execute_on_epoch_end() for hook in self.hooks]
            self.context.batch_idx = 0

        self.context.batch_idx = self.context.num_steps_per_epoch - 1
        # Execute hooks on training end
        [hook.execute_on_training_end() for hook in self.hooks]
        # Wait for everyone to finish
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


    def step(self, batch: Any) -> Tensor:
        # Set models to training mode
        for model in self.models:
            model.train()
        # Wrap forward and backward passes to enable gradient accumulation
        with self.accelerator.accumulate(*self.models):
            # Zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            # Forward pass with mixed precision
            with self.accelerator.autocast():
                loss = self.compute_loss(batch)
            # Backward pass
            self.accelerator.backward(loss)
            # Gradient clipping
            self._clip_gradients()
            # Update parameters
            for optimizer in self.optimizers:
                optimizer.step()
        # Return loss for context tracking (detach to prevent memory leaks)
        return loss.detach()


    def compute_loss(self, batch: Any) -> Tensor:
        raise NotImplementedError


    def _clip_gradients(self) -> None:
        if not self.accelerator.sync_gradients:
            return
        for model in self.models:
            if self.gradient_clipping_value is not None:
                self.accelerator.clip_grad_value_(
                    parameters=model.parameters(),
                    clip_value=self.gradient_clipping_value,
                )
            if self.gradient_clipping_norm is not None:
                self.accelerator.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=self.gradient_clipping_norm,
                )


    def __repr__(self) -> str:
        result = f"{self.__class__.__name__}(\n\t"
        result += ",\n\t".join(
            [f"{hook.name}: {hook}" for hook in self.hooks]
        )
        result += "\n)"
        return result
