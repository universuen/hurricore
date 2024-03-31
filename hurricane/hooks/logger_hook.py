import time
from logging import Logger

from torch.cuda import memory_reserved

from hurricane.hooks import HookBase
from hurricane.trainers import TrainerBase
from hurricane.utils import (
    auto_name,
    get_list_mean,
    get_total_parameters,
    get_trainable_parameters,
)


class LoggerHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        logger: Logger = None,
        interval: int = 1, 
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert interval > 0, 'Log interval must be greater than 0.'
        assert logger is not None, 'Invalid logger.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        # setup self
        self.logger = logger
        self.interval = interval
    
    
    def on_training_start(self) -> None:
        if self.trainer.accelerator.is_main_process:
            models = self.trainer.originals.models
            for name, model in zip(auto_name(models), models):
                self.logger.info(f'{name} structure:\n{model}')
                self.logger.info(f'{name} total parameters: {get_total_parameters(model)}')
                self.logger.info(f'{name} trainable parameters: {get_trainable_parameters(model)}')
    
    
    def on_epoch_start(self) -> None:
        self.step = 0
        if self.trainer.accelerator.is_main_process:
            self.step_losses = []
            self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} started')
            self.start_time = time.time()
        
        
    def on_step_end(self) -> None:
        self.step += 1
        if (self.trainer.ctx.global_step + 1) % self.interval == 0:
            self._collect_step_loss()
            if self.trainer.accelerator.is_main_process:
                self._log_states()
      
      
    def on_epoch_end(self) -> None: 
        if self.trainer.accelerator.is_main_process:
            self._log_states()
            avg_loss = get_list_mean(self.step_losses)
            self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} finished with average loss: {avg_loss: .5f}')


    def _collect_step_loss(self):
        step_loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
        if self.trainer.accelerator.is_main_process:
            self.step_losses.append(step_loss)
    
    
    def _get_remaining_time(self):
        elapsed_time = time.time() - self.start_time
        iterator_length = self.trainer.ctx.iterator_length
        batches_idx = self.trainer.ctx.batches_idx + 1
        remaining_time = (iterator_length - batches_idx) * (elapsed_time / self.step)
        days, remainder = divmod(remaining_time, 86400) 
        hours, remainder = divmod(remainder, 3600) 
        minutes, seconds = divmod(remainder, 60) 
        formatted_remaining_time = f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return formatted_remaining_time
    
    
    def _log_states(self):
        idx = self.trainer.ctx.batches_idx + 1
        iterator_length = self.trainer.ctx.iterator_length
        epoch = self.trainer.ctx.epoch + 1
        progress = idx / iterator_length
        remaining_time = self._get_remaining_time()
        
        self.logger.info(
            f"Epoch: {epoch}/{self.trainer.epochs} | "
            f"Step: {idx}/{iterator_length} | "
            f"Loss: {self.step_losses[-1]:.5f} | "
            f"Progress: {progress:.2%} | "
            f"Time left: {remaining_time} | "
            f"Memory used: {memory_reserved() / 1024 ** 3:.2f}GB"
        )
    