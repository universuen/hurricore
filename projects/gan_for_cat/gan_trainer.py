from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers import Trainer
from hurricane.hooks import CheckpointHook

from hooks import GANLoggerHook, ImgPeekHook, GANTensorBoardHook
from models import Generator, Discriminator


def _compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size()).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class GANTrainer(Trainer):
    def __init__(
        self, 
        
        g_model: Generator,
        d_model: Discriminator,
        
        data_loader: DataLoader, 
        
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        
        accelerator: Accelerator, 
        epochs: int = 100,
        
        g_loop_per_step: int = 1,
        d_loop_per_step: int = 1,
        
        lambda_gp: int = 10,
        
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
            epochs=epochs,
        )
        
        self.g_loop_per_step = g_loop_per_step
        self.d_loop_per_step = d_loop_per_step
        self.lambda_gp = lambda_gp
        
        self.hooks = [
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
                    fake_images = g_model(z)
                    # compute scores
                    real_scores = d_model(real_images)
                    fake_scores = d_model(fake_images)
                    # compute loss
                    d_loss = (fake_scores.mean() - real_scores.mean()) / 2
                self.accelerator.backward(d_loss)
                # clip parameters
                with torch.no_grad():
                    for p in d_model.parameters():
                        p.grad.data.clamp_(-0.1, 0.1)
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
