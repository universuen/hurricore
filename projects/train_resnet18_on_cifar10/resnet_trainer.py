import path_setup

from logging import Logger
from pathlib import Path

from accelerate import Accelerator
from torch.nn.functional import cross_entropy
from torch import Tensor
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.hooks.ckpt_hook import CKPTHook


class ResNetTrainer(Trainer):
    def __init__(
        self, 
        model: Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        logger: Logger,
        ckpt_folder_path: Path = None
    ) -> None:
        super().__init__(
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            accelerator=accelerator
        )
        self.hooks = [
            LoggerHook(logger=logger),
            CKPTHook(folder_path=ckpt_folder_path),
        ]
        
    def compute_loss(self) -> Tensor:
        inputs, labels = self.ctx.batch
        outputs = self.model(inputs)
        loss = cross_entropy(outputs, labels)
        return loss
