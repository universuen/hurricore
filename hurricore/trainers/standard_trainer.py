from __future__ import annotations

import os
import math
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

from accelerate import Accelerator
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader

from hurricore.trainers import HurricoreTrainer
from hurricore.hooks import (
    LoggingHook, 
    LRSchedulerHook, 
    TensorBoardHook, 
    HistoryHook,
    ValidationLossHook,
    CheckpointHook,
)
from hurricore.utils import AttributeDict, HurricoreHookList


class StandardTrainer(HurricoreTrainer):
    def __init__(
        self, 
        # Basic configs
        accelerator: Accelerator,
        model: nn.Module, 
        data_folder_path: Path | str,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        collate_fn: Callable | None = None,
        num_epochs: int = 3,
        per_device_batch_size: int = 1,
        learning_rate: float = 5e-5,
        gradient_clipping_norm: float | None = None,
        gradient_clipping_value: float | None = None,
        warmup_ratio: float = 0.1,
        # Hook specific configs
        experiment_name: str | None = None,
        log_level: int = logging.INFO,
        log_interval: int = 1,
        log_trainable_params: bool = False,
        tensorboard_record_param_and_grad: bool = False,
        evaluation_interval: int = 100,
        ckpt_interval: int = 500,
        **kwargs,
    ) -> None:
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M')
        if type(data_folder_path) is not Path:
            data_folder_path = Path(data_folder_path)
        data_folder_path.mkdir(parents=True, exist_ok=True)
        
        self.init_context = AttributeDict()
        self.init_context.update({k: v for k, v in locals().items() if k not in ("self", "kwargs")} | kwargs)
        self.setup_data_loaders()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_hooks()
        self._attach_to_hurricore_trainer()


    def setup_data_loaders(self, num_workers: int = None) -> None:
        if num_workers is None:
            num_workers = os.cpu_count() // self.init_context.accelerator.num_processes
        self.init_context.training_data_loader = DataLoader(  # noqa
            dataset=self.init_context.training_dataset,
            batch_size=self.init_context.per_device_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.init_context.collate_fn,
        )
        self.init_context.validation_data_loader = DataLoader(  # noqa
            dataset=self.init_context.validation_dataset,
            batch_size=self.init_context.per_device_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.init_context.collate_fn,
        )


    def setup_optimizer(self) -> None:
        self.init_context.optimizer = AdamW(  # noqa
            params=self.init_context.model.parameters(),
            lr=self.init_context.learning_rate,
        )

    
    def setup_lr_scheduler(self) -> None:
        num_steps_per_epoch = math.ceil(len(self.init_context.training_data_loader) / self.init_context.accelerator.num_processes)
        num_training_steps = math.ceil(num_steps_per_epoch * self.init_context.num_epochs / self.init_context.accelerator.gradient_accumulation_steps)
        self.init_context.lr_scheduler = get_cosine_schedule_with_warmup(  # noqa
            optimizer=self.init_context.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=int(num_training_steps * self.init_context.warmup_ratio),
        )
        

    def setup_hooks(self) -> None:
        self.init_context.hooks = HurricoreHookList(
            [
                LoggingHook(
                    trainer=self,
                    experiment_name=self.init_context.experiment_name,
                    level=self.init_context.log_level,
                    log_trainable_params=self.init_context.log_trainable_params,
                    log_file_path=self.init_context.data_folder_path / 'logs' / f'{self.init_context.experiment_name}.log',
                    interval=self.init_context.log_interval,
                ),
                LRSchedulerHook(
                    trainer=self,
                    lr_schedulers=[self.init_context.lr_scheduler],
                    modes=['per_step'],
                ),
                TensorBoardHook(
                    trainer=self,
                    folder_path=self.init_context.data_folder_path / 'tensorboards' / self.init_context.experiment_name,
                    record_param_and_grad=self.init_context.tensorboard_record_param_and_grad,
                    interval=self.init_context.log_interval,
                ),
                HistoryHook(
                    trainer=self,
                    folder_path=self.init_context.data_folder_path / 'histories' / self.init_context.experiment_name,
                    interval=self.init_context.log_interval,
                ),
                ValidationLossHook(
                    trainer=self,
                    data_loader=self.init_context.validation_data_loader,
                    interval=self.init_context.evaluation_interval,
                ),
                CheckpointHook(
                    trainer=self,
                    folder_path=self.init_context.data_folder_path / 'checkpoints' / self.init_context.experiment_name,
                    interval=self.init_context.ckpt_interval,
                ),
            ]
        )


    def _attach_to_hurricore_trainer(self) -> None:
        super().__init__(
            models=[self.init_context.model], 
            data_loaders=[self.init_context.training_data_loader], 
            optimizers=[self.init_context.optimizer], 
            accelerator=self.init_context.accelerator,
            num_epochs=self.init_context.num_epochs,
            gradient_clipping_norm=self.init_context.gradient_clipping_norm,
            gradient_clipping_value=self.init_context.gradient_clipping_value,
            hooks=self.init_context.hooks,
        )


    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError
