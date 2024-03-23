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
from hurricane.hooks.lr_scheduler_hook import LRSchedulerHook
from hurricane.hooks.tensor_board_hook import TensorBoardHook
from hurricane.hooks.ckpt_hook import CKPTHook


class ResNetTrainer(Trainer):
    def __init__(
        self, 
        model: Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        epochs: int = 100,
        seed: int = 42,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
        
        tensorboard_folder_path: Path = None,
        tensorboard_interval: int = 1,
        
        ckpt_folder_path: Path = None,
        ckpt_interval: int = 100,
        
    ) -> None:
        super().__init__(
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            accelerator=accelerator,
            epochs=epochs,
            seed=seed,
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
            TensorBoardHook(
                folder_path=tensorboard_folder_path,
                interval=tensorboard_interval,    
            ),
            CKPTHook(
                folder_path=ckpt_folder_path,
                interval=ckpt_interval,
            ),
        ]
        
    def compute_loss(self) -> Tensor:
        from time import sleep
        sleep(1)
        inputs, labels = self.ctx.batch
        outputs = self.model(inputs)
        loss = cross_entropy(outputs, labels)
        return loss
