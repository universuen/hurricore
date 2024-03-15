from logging import Logger
import time

from torch.cuda import memory_cached

from src.hooks.logger_hook import LoggerHook
from src.trainers.trainer import Trainer


class LoggerHookWithAccelerator(LoggerHook):
    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
    
    def iteration_end(self, trainer: Trainer) -> None:
        step_loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()

        if trainer.accelerator.is_main_process:
            self.losses_per_batch.append(step_loss)
        idx = trainer.ctx.batch_idx + 1
        num_batches = len(trainer.data_loader)
        if trainer.accelerator.is_main_process and (idx % 10 == 0 or idx == num_batches):
            progress = (idx / num_batches)
            elapsed_time = time.time() - self.start_time
            remaining_time = (elapsed_time / idx) * (num_batches - idx)
            formatted_remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            self.logger.info(
                f"Epoch: {self.epoch}/{self.epochs} | "
                f"Step: {idx}/{num_batches} | "
                f"Loss: {step_loss:.5f} | "
                f"Progress: {progress:.2%} | "
                f"Time left: {formatted_remaining_time} | "
                f"Memory cached: {memory_cached() / 1024 ** 3:.2f}GB"
            )

            
