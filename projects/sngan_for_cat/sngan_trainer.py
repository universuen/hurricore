from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.checkpoint_hook import CheckpointHook

from hooks.sngan_logger_hook import SNGANLoggerHook
from hooks.sngan_tensor_baord_hook import SNGANTensorBoardHook
from hooks.img_peek_hook import ImgPeekHook
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
        g_loop_per_step: int = 1,
        d_loop_per_step: int = 1,
        seed: int = 42,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        tensor_board_folder_path: str = None,
        tensor_board_interval: int = 1,
        
        image_peek_folder_path: str = None,
        image_peek_interval: int = 1,
        
        checkpoint_folder_path: str = None,
        checkpoint_interval: int = 1000,
        
    ) -> None:
        super().__init__(model, data_loader, None, accelerator, epochs, seed)
        self.g_loop_per_step = g_loop_per_step
        self.d_loop_per_step = d_loop_per_step
        
        g_optimizer, d_optimizer = accelerator.prepare(g_optimizer, d_optimizer)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.hooks = [
            SNGANLoggerHook(
                trainer=self,
                logger=logger,
                interval=log_interval,
            ),
            ImgPeekHook(
                trainer=self,
                folder_path=image_peek_folder_path,
                interval=image_peek_interval,
            ),
            SNGANTensorBoardHook(
                trainer=self,
                folder_path=tensor_board_folder_path,
                interval=tensor_board_interval,
            ),
            CheckpointHook(
                trainer=self,
                folder_path=checkpoint_folder_path,
                interval=checkpoint_interval,
            ),
        ]

    def training_step(self) -> torch.Tensor:
        self.model.train()
        
        with self.accelerator.accumulate(self.model):
            
            batch_size = self.ctx.batch.shape[0]
            
            for _ in range(self.d_loop_per_step):
                self.d_optimizer.zero_grad()
                with self.accelerator.autocast():
                    fake_images = self.model.generator.generate(batch_size)
                    real_images = self.ctx.batch
                    real_scores = self.model.discriminator(real_images)
                    fake_scores = self.model.discriminator(fake_images)
                    avg_real_score = real_scores.mean()
                    avg_fake_score = fake_scores.mean()
                    d_loss = (avg_fake_score - avg_real_score) / 2
                self.accelerator.backward(d_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.discriminator.parameters(), 1)
                self.d_optimizer.step()
    
            for _ in range(self.g_loop_per_step):
                with self.accelerator.autocast():
                    fake_images = self.model.generator.generate(batch_size)
                    fake_scores = self.model.discriminator(fake_images)
                    avg_fake_score = fake_scores.mean()
                    g_loss = -avg_fake_score
                self.accelerator.backward(g_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.generator.parameters(), 1)
                self.g_optimizer.step()
                self.g_optimizer.zero_grad()
            
            
            
            self.ctx.g_step_loss = g_loss
            self.ctx.d_step_loss = d_loss
            
            return torch.tensor([0])
