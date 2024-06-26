from logging import Logger
from pathlib import Path

from accelerate import Accelerator
from torch.nn.functional import cross_entropy
from torch import Tensor
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader

from hurricore.trainers import Trainer
from hurricore.hooks import (
    LoggerHook, 
    LRSchedulerHook, 
    TensorBoardHook, 
    CheckpointHook,
)


class ResNetTrainer(Trainer):
    def __init__(
        self, 
        model: Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        num_epochs: int = 100,
        ckpt_seed: int = 42,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
        
        tensor_board_folder_path: Path = None,
        tensor_board_interval: int = 1,
        
        ckpt_folder_path: Path = None,
        ckpt_interval: int = 100,
        
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
            ),
            CheckpointHook(
                trainer=self,
                folder_path=ckpt_folder_path,
                interval=ckpt_interval,
                seed=ckpt_seed,
            ),
        ]
        
    def compute_loss(self) -> Tensor:
        inputs, labels = self.ctx.batches[0]
        model = self.models[0]
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        return loss
