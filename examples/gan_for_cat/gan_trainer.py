from logging import Logger

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers import Trainer
from hurricane.hooks import CheckpointHook, LRSchedulerHook, SyncBatchNormHook

from hooks import GANLoggerHook, ImgPeekHook, GANTensorBoardHook
from models import Generator, Discriminator


def _compute_gradient_penalty(
        d_model: torch.nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        device: torch.device,
) -> torch.Tensor:
    alpha = torch.rand(real_images.shape[0], 1, 1, 1).to(device)

    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad = True

    disc_interpolates = d_model(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size()[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class GANTrainer(Trainer):
    def __init__(
        self, 
        
        data_loader: DataLoader, 
        accelerator: Accelerator, 
        
        g_model: Generator,
        d_model: Discriminator,
        
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        
        num_epochs: int = 100,
        gp_lambda: float = 10.0,
        
        g_lr_scheduler: LRScheduler = None,
        d_lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
        
        g_loop_per_step: int = 1,
        d_loop_per_step: int = 1,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        tensor_board_folder_path: str = None,
        tensor_board_interval: int = 1,
        
        image_peek_folder_path: str = None,
        image_peek_interval: int = 1,
        
        checkpoint_folder_path: str = None,
        checkpoint_interval: int = 1000,
        checkpoint_seed: int = 42,
    ) -> None:
        
        super().__init__(
            models=[g_model, d_model], 
            data_loaders=[data_loader], 
            optimizers=[g_optimizer, d_optimizer], 
            accelerator=accelerator,
            num_epochs=num_epochs,
        )
        
        self.gp_lambda = gp_lambda
        
        self.g_loop_per_step = g_loop_per_step
        self.d_loop_per_step = d_loop_per_step
        
        self.hooks = [
            SyncBatchNormHook(
                trainer=self
            ),
            GANLoggerHook(
                trainer=self,
                logger=logger,
                interval=log_interval,
            ),
            ImgPeekHook(
                trainer=self,
                folder_path=image_peek_folder_path,
                interval=image_peek_interval,
            ),
            GANTensorBoardHook(
                trainer=self,
                folder_path=tensor_board_folder_path,
                interval=tensor_board_interval,
            ),
            LRSchedulerHook(
                trainer=self,
                lr_schedulers=[g_lr_scheduler, d_lr_scheduler],
                mode=lr_scheduler_mode,
            ),
            CheckpointHook(
                trainer=self,
                folder_path=checkpoint_folder_path,
                interval=checkpoint_interval,
                seed=checkpoint_seed,
            ),
        ]

    def training_step(self) -> torch.Tensor:
        for model in self.models:
            model.train()
            
        with self.accelerator.accumulate(*self.models):
            real_images = self.ctx.batches[0]
            batch_size = real_images.size(0)
            g_model, d_model = self.models
            z_dim = self.originals.models[0].z_dim
            g_optimizer, d_optimizer = self.optimizers
            device = self.accelerator.device
            # train discriminator
            for _ in range(self.d_loop_per_step):
                d_optimizer.zero_grad()
                with self.accelerator.autocast():
                    # construct fake images
                    z = torch.randn(batch_size, z_dim).to(device)
                    with torch.no_grad():
                        fake_images = g_model(z).detach()
                    # compute scores
                    real_scores = d_model(real_images)
                    fake_scores = d_model(fake_images)
                    # compute gradient penalty
                    gradient_penalty = _compute_gradient_penalty(d_model, real_images, fake_images, device)
                    # compute loss with parameter regularization
                    d_loss = (fake_scores.mean() - real_scores.mean()) / 2 + self.gp_lambda * gradient_penalty
                self.accelerator.backward(d_loss)
                d_optimizer.step()
            # train generator
            for _ in range(self.g_loop_per_step):
                g_optimizer.zero_grad()
                with self.accelerator.autocast():
                    # construct fake images
                    z = torch.randn(batch_size, z_dim).to(device)
                    fake_images = g_model(z)
                    # compute scores
                    fake_scores = d_model(fake_images)
                    # compute loss
                    avg_fake_score = fake_scores.mean()
                    g_loss = -avg_fake_score
                self.accelerator.backward(g_loss)
                g_optimizer.step()
            # save losses to context
            self.ctx.g_step_loss = g_loss
            self.ctx.d_step_loss = d_loss
            # return dummy loss
            return torch.tensor([.0], device=self.accelerator.device)
