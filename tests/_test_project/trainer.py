from __future__ import annotations

import os
import logging
from pathlib import Path

from accelerate import Accelerator
import torch
from torch.nn.modules import Module
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from hurricore.trainers import HurricoreTrainer
from hurricore.hooks import (
    LoggingHook, 
    LRSchedulerHook, 
    TensorBoardHook, 
    HistoryHook,
    CheckpointHook,
    ValidationLossHook,
)

from hooks.my_cls_metric_hook import MyClsEvaluationHook


class TestTrainer(HurricoreTrainer):
    def __init__(
        self, 
        # Basic configs
        accelerator: Accelerator,
        model: Module, 
        dataset: Dataset,
        data_folder_path: Path | str,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        # Hook specific configs
        experiment_name: str = None,
        log_level: int = logging.INFO,
        log_interval: int = 1,
        tensor_board_record_param_and_grad: bool = False,
        evaluation_interval: int = 100,
        ckpt_interval: int = 500,
    ) -> None:

        # setup data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2,
        )

        # setup optimizer
        optimizer = AdamW(
            params=model.parameters(),
            lr=lr,
        )

        # setup lr scheduler
        num_steps_per_epoch = len(data_loader)
        gradient_accumulation_steps = accelerator.gradient_accumulation_steps
        num_training_steps = num_steps_per_epoch * num_epochs // gradient_accumulation_steps
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps,
        )

        # integrate all components
        super().__init__(
            models=[model], 
            data_loaders=[data_loader], 
            optimizers=[optimizer], 
            accelerator=accelerator,
            num_epochs=num_epochs,
        )

        # setup hooks
        self.hooks = [
            LoggingHook(
                trainer=self,
                experiment_name=experiment_name,
                level=log_level,
                folder_path=data_folder_path / 'logs',
                interval=log_interval,
            ),
            LRSchedulerHook(
                trainer=self,
                lr_schedulers=[lr_scheduler],
                modes=['per_step'],
            ),
            TensorBoardHook(
                trainer=self,
                folder_path=data_folder_path / 'tensorboards' / experiment_name,
                record_param_and_grad=tensor_board_record_param_and_grad,
            ),
            HistoryHook(
                trainer=self,
                folder_path=data_folder_path / 'histories' / experiment_name,
            ),
            CheckpointHook(
                trainer=self,
                folder_path=data_folder_path / 'checkpoints' / experiment_name,
                interval=ckpt_interval,
            ),
            MyClsEvaluationHook(
                trainer=self,
                task='multiclass',
                num_classes=10,
                average='macro',
                data_loader=data_loader,
                interval=10,
            ),
            ValidationLossHook(
                trainer=self,
                data_loader=data_loader,
                interval=10,
            )
        ]


    def compute_loss(self) -> torch.Tensor:
        x, y = self.context.batch
        y_pred = self.model(x)
        return torch.nn.functional.cross_entropy(y_pred, y)
