import time
from logging import Logger

from torch.cuda import memory_reserved

from hurricane.hooks import HookBase
from hurricane.trainers import TrainerBase
from hurricane.utils import get_list_mean


def _format_parameters(num_params):
    if num_params >= 1e9: 
        return f'{num_params / 1e9:.2f} B'
    elif num_params >= 1e6: 
        return f'{num_params / 1e6:.2f} M'
    elif num_params >= 1e3: 
        return f'{num_params / 1e3:.2f} K'
    else: 
        return str(num_params)


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
            model = self.trainer.originals.model
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.logger.info(f'Model structure:\n{model}')
            self.logger.info(f'Total parameters: {_format_parameters(total_params)}')
            self.logger.info(f'Trainable parameters: {_format_parameters(trainable_params)}')
    
    def on_epoch_start(self) -> None:
        self.step = 0
        if self.trainer.accelerator.is_main_process:
            assert hasattr(self.trainer.ctx, 'epoch')
            self.losses_per_batch = []
            self.logger.info(f'Epoch {self.trainer.ctx.epoch} started')
            self.start_time = time.time()
            
    
    def _get_remaining_time(self, num_batches, idx):
        elapsed_time = time.time() - self.start_time
        remaining_time = (num_batches - idx) * (elapsed_time / self.step)
        days, remainder = divmod(remaining_time, 86400) 
        hours, remainder = divmod(remainder, 3600) 
        minutes, seconds = divmod(remainder, 60) 
        formatted_remaining_time = f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return formatted_remaining_time
        
    def on_step_end(self) -> None:
        self.step += 1
        idx = self.trainer.ctx.batch_idx
        num_batches = len(self.trainer.data_loader)
        if self.trainer.ctx.global_step % self.interval == 0 or idx == num_batches:
            step_loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                self.losses_per_batch.append(step_loss)
                epoch = self.trainer.ctx.epoch
                progress = idx / num_batches
                remaining_time = self._get_remaining_time(num_batches, idx)
                
                self.logger.info(
                    f"Epoch: {epoch}/{self.trainer.epochs} | "
                    f"Step: {idx}/{num_batches} | "
                    f"Loss: {step_loss:.5f} | "
                    f"Progress: {progress:.2%} | "
                    f"Time left: {remaining_time} | "
                    f"Memory used: {memory_reserved() / 1024 ** 3:.2f}GB"
                )
                
    def on_epoch_end(self) -> None: 
        if self.trainer.accelerator.is_main_process:
            avg_loss = get_list_mean(self.losses_per_batch)
            self.logger.info(f'Epoch {self.trainer.ctx.epoch} finished with average loss: {avg_loss}')
