import _path_setup  # noqa: F401

from logging import Logger
from pathlib import Path

from accelerate import Accelerator
from torch import Tensor
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader

from hurricane.trainers import Trainer
from hurricane.hooks import (
    LoggerHook, 
    LRSchedulerHook, 
    TensorBoardHook, 
    CheckpointHook,
)


class MyTrainer(Trainer):
    def __init__(
        # basic configs
        self, 
        model: Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        num_epochs: int = 100,
        # logger hook
        logger: Logger = None,
        log_interval: int = 1,
        # lr scheduler hook
        lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
        # tensor board hook
        tensor_board_folder_path: Path = None,
        tensor_board_interval: int = 1,
        tensor_board_record_grad: bool = False,
        # checkpoint hook
        ckpt_folder_path: Path = None,
        ckpt_interval: int = 100,
        ckpt_seed: int = 42,
    ) -> None:
        super().__init__(
            models=[model], 
            data_loaders=[data_loader], 
            optimizers=[optimizer], 
            accelerator=accelerator,
            num_epochs=num_epochs,
        )
        self.hooks = [
            LoggerHook(
                trainer=self,
                logger=logger,
                interval=log_interval,
            ),
            LRSchedulerHook(
                trainer=self,
                lr_schedulers=[lr_scheduler],
                mode=lr_scheduler_mode,
            ),
            TensorBoardHook(
                trainer=self,
                folder_path=tensor_board_folder_path,
                interval=tensor_board_interval,
                record_grad=tensor_board_record_grad,
            ),
            CheckpointHook(
                trainer=self,
                folder_path=ckpt_folder_path,
                interval=ckpt_interval,
                seed=ckpt_seed,
            ),
        ]
        
    def compute_loss(self) -> Tensor:
        ...
