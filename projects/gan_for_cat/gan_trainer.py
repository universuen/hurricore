from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.checkpoint_hook import CheckpointHook

from hooks.gan_logger_hook import GANLoggerHook
from hooks.gan_tensor_baord_hook import GANTensorBoardHook
from hooks.img_peek_hook import ImgPeekHook
from projects.gan_for_cat.gan import GAN


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
        model: GAN,
        data_loader: DataLoader, 
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        accelerator: Accelerator, 
        epochs: int = 100,
        
        g_loop_per_step: int = 1,
        d_loop_per_step: int = 1,
        
        lambda_gp: int = 10,
        
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
        self.lambda_gp = lambda_gp
        g_model, d_model = accelerator.prepare(model.generator, model.discriminator)
        self.g_model = g_model
        self.d_model = d_model
        g_optimizer, d_optimizer = accelerator.prepare(g_optimizer, d_optimizer)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.z_dim = model.z_dim
        
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
            ),
        ]

    def training_step(self) -> torch.Tensor:
        self.model.train()
        
        with self.accelerator.accumulate(self.model):
            
            batch_size = self.ctx.batch.shape[0]

            for _ in range(self.d_loop_per_step):
                with self.accelerator.autocast():
                    z = torch.randn(batch_size, self.z_dim, 1, 1).to(self.ctx.batch.device)
                    fake_images = self.g_model(z).detach()
                    real_images = self.ctx.batch
                    real_scores = self.d_model(real_images)
                    fake_scores = self.d_model(fake_images)
                    gradient_penalty = _compute_gradient_penalty(self.d_model, real_images, fake_images)
                    d_loss = (fake_scores.mean() - real_scores.mean()) + self.lambda_gp * gradient_penalty
                self.accelerator.backward(d_loss)
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
            
            for _ in range(self.g_loop_per_step):
                with self.accelerator.autocast():
                    z = torch.randn(batch_size, self.z_dim, 1, 1).to(self.ctx.batch.device)
                    fake_images = self.g_model(z)
                    fake_scores = self.d_model(fake_images)
                    avg_fake_score = fake_scores.mean()
                    g_loss = -avg_fake_score
                self.accelerator.backward(g_loss)
                self.g_optimizer.step()
                self.g_optimizer.zero_grad()
            
            self.ctx.g_step_loss = g_loss
            self.ctx.d_step_loss = d_loss
            
            return torch.tensor([0])
