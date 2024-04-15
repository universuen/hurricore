from torch import Tensor
import _path_setup  # noqa: F401

from logging import Logger
from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator

from hurricane.trainers import Trainer
from hurricane.hooks import (
    LoggerHook,
    LRSchedulerHook,
    TensorBoardHook,
    CheckpointHook,
)

from img_peek_hook import ImgPeekHook
from noise_schedulers import DDPMNoiseScheduler


class DiffusionTrainer(Trainer):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            data_loader: DataLoader,
            accelerator: Accelerator,
            num_epochs: int,
            
            noise_scheduler: DDPMNoiseScheduler,
            
            lr_scheduler: LRScheduler = None,
            lr_scheduler_mode: str = 'per_epoch',
            
            logger: Logger = None,
            log_interval: int = 1,
            
            tensor_board_folder_path: str = None,
            tensor_board_interval: int = 1,
            
            ckpt_folder_path: Path = None,
            ckpt_interval: int = 1000,
            ckpt_seed: int = 42,
            
        ):
        super().__init__(
            models=[model],
            optimizers=[optimizer],
            data_loaders=[data_loader],
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
        self.noise_scheduler = noise_scheduler
        
    def compute_loss(self) -> Tensor:
        model = self.models[0]
        batch = self.ctx.batches[0]
        t = torch.randint(self.noise_scheduler.num_steps)
        corrupted_images, noise = self.noise_scheduler.corrupt(batch, t)
        predicted_noise = model(corrupted_images, t)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        return loss
