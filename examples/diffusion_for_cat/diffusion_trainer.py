from torch import Tensor

from logging import Logger
from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator

from hurricore.trainers import Trainer
from hurricore.hooks import (
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
            
            image_peek_folder_path: Path = None,
            image_peek_interval: int = 1000,
            
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
        
        noise_scheduler = noise_scheduler.to(self.accelerator.device)
        self.noise_scheduler = noise_scheduler
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
            ImgPeekHook(
                trainer=self,
                folder_path=image_peek_folder_path,
                interval=image_peek_interval,
            ),
            CheckpointHook(
                trainer=self,
                folder_path=ckpt_folder_path,
                interval=ckpt_interval,
                seed=ckpt_seed,
            ),
        ]
        
        
    def compute_loss(self) -> Tensor:
        model = self.models[0]
        batch = self.ctx.batches[0]
        t = torch.randint(0, self.noise_scheduler.num_steps, (batch.shape[0],)).to(batch.device)
        corrupted_images, noise = self.noise_scheduler.corrupt(batch, t)
        predicted_noise = model(corrupted_images, t)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        return loss
