import time
from logging import Logger

from torch.cuda import memory_reserved

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer
from hurricane.trainers.trainer_base import TrainerBase
from hurricane.utils import get_list_mean


class LoggerHook(HookBase):
    def __init__(
        self, 
        logger: Logger = None,
        interval: int = 1, 
    ) -> None:
        super().__init__()
        self.is_available = (logger is not None)
        self.logger = logger
        self.interval = interval
    
    def on_training_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        trainer.logger = self.logger
    
    def on_epoch_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        if trainer.accelerator.is_main_process:
            assert hasattr(trainer.ctx, 'epoch')
            self.losses_per_batch = []
            self.logger.info(f'Epoch {trainer.ctx.epoch} started')
            self.start_time = time.time()
        
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        step_loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()
        if trainer.accelerator.is_main_process:
            self.losses_per_batch.append(step_loss)
            idx = trainer.ctx.batch_idx + 1
            epoch = trainer.ctx.epoch
            num_batches = len(trainer.data_loader)
            if idx % self.interval == 0 or idx == num_batches:
                progress = idx / num_batches
                elapsed_time = time.time() - self.start_time
                remaining_time = (elapsed_time / idx) * ((trainer.epochs - epoch + 1) * num_batches - idx)
                days, remainder = divmod(remaining_time, 86400) 
                hours, remainder = divmod(remainder, 3600) 
                minutes, seconds = divmod(remainder, 60) 
                formatted_remaining_time = f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                self.logger.info(
                    f"Epoch: {epoch}/{trainer.epochs} | "
                    f"Step: {idx}/{num_batches} | "
                    f"Loss: {step_loss:.5f} | "
                    f"Progress: {progress:.2%} | "
                    f"Time left: {formatted_remaining_time} | "
                    f"Current lr: {trainer.optimizer.param_groups[0]['lr']} | "
                    f"Memory used: {memory_reserved() / 1024 ** 3:.2f}GB"
                )
                

    def on_epoch_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        if trainer.accelerator.is_main_process:
            avg_loss = get_list_mean(self.losses_per_batch)
            self.logger.info(f'Epoch {trainer.ctx.epoch} finished with average loss: {avg_loss}')

