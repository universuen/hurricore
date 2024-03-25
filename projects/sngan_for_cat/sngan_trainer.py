from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers.trainer import Trainer
from hooks.sngan_logger_hook import SNGANLoggerHook
from hooks.sngan_tensor_baord_hook import SNGANTensorBoardHook

from sngan import SNGAN


class SNGANTrainer(Trainer):
    def __init__(
        self, 
        model: SNGAN,
        data_loader: DataLoader, 
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        accelerator: Accelerator, 
        epochs: int = 100, 
        seed: int = 42,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        tensor_board_folder_path: str = None,
        tensor_board_interval: int = 1,
        
    ) -> None:
        super().__init__(model, data_loader, None, accelerator, epochs, seed)
        g_optimizer, d_optimizer = accelerator.prepare(g_optimizer, d_optimizer)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.hooks = [
            SNGANLoggerHook(
                logger=logger,
                interval=log_interval,
            ),
            SNGANTensorBoardHook(
                folder_path=tensor_board_folder_path,
                interval=tensor_board_interval,
            ),
        ]

    def training_step(self) -> torch.Tensor:
        self.model.train()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                g_loss, d_loss = self.model(self.ctx.batch)
                
            self.g_optimizer.zero_grad()
            self.accelerator.backward(g_loss)
            self.g_optimizer.step()
            
            self.d_optimizer.zero_grad()
            self.accelerator.backward(d_loss)
            self.d_optimizer.step()
            
            self.ctx.g_step_loss = g_loss
            self.ctx.d_step_loss = d_loss
            
            return torch.tensor([0])
