import _path_setup  # noqa: F401

from logging import Logger
from pathlib import Path

from accelerate import Accelerator
import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import Dataset, DataLoader

from hurricane.trainers import Trainer
from hurricane.hooks import (
    LoggerHook, 
    LRSchedulerHook, 
    TensorBoardHook, 
    CheckpointHook,
)

from img_peek_hook import ImgPeekHook


class FlowTrainer(Trainer):
    def __init__(
        self, 
        model: Module, 
        training_data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        num_epochs: int = 100,
        
        img_peek_dataset: Dataset = None,
        img_peek_folder_path: Path = None,
        img_peek_interval: int = 10,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
        
        tensor_board_folder_path: Path = None,
        tensor_board_interval: int = 1,
        
        ckpt_folder_path: Path = None,
        ckpt_interval: int = 100,
        ckpt_seed: int = 42,
    ) -> None:
        super().__init__(
            models=[model], 
            data_loaders=[training_data_loader], 
            optimizers=[optimizer], 
            accelerator=accelerator,
            num_epochs=num_epochs,
        )
        self.hooks = [
            ImgPeekHook(
                trainer=self,
                dataset=img_peek_dataset,
                folder_path=img_peek_folder_path,
                interval=img_peek_interval,
            ),
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
        model = self.models[0]
        cat_images, dog_images = self.ctx.batches[0]
        batch_size = cat_images.shape[0]
        expected_velocities = dog_images - cat_images
        time = torch.rand((batch_size, ), device=self.accelerator.device)
        positions = cat_images + time.reshape(-1, 1, 1, 1) * expected_velocities
        predicted_velocities = model(positions, time)
        loss = ((predicted_velocities - expected_velocities) ** 2).mean()
        return loss
