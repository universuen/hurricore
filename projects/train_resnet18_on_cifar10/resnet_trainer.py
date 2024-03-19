import path_setup

from logging import Logger
from pathlib import Path

from accelerate import Accelerator
from torch.nn.functional import cross_entropy
from torch import Tensor
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.hooks.ckpt_hook import CKPTHook
from hurricane.hooks.lr_scheduler_hook import LRSchedulerHook


class ResNetTrainer(Trainer):
    def __init__(
        self, 
        model: Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        logger: Logger,
        log_interval: int = 1,
        ckpt_folder_path: Path = None,
        lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
    ) -> None:
        super().__init__(
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            accelerator=accelerator
        )
        self.hooks = [
            LoggerHook(
                logger=logger,
                interval=log_interval,
            ),
            LRSchedulerHook(
                lr_scheduler=lr_scheduler,
                mode=lr_scheduler_mode,
            ),
            CKPTHook(folder_path=ckpt_folder_path),
        ]
        
    def compute_loss(self) -> Tensor:
        inputs, labels = self.ctx.batch
        outputs = self.model(inputs)
        loss = cross_entropy(outputs, labels)
        return loss
