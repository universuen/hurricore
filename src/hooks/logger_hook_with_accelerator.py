import time
from logging import Logger

from torch.cuda import memory_cached

from src.hooks.logger_hook import LoggerHook
from src.trainers.trainer import Trainer


class LoggerHookWithAccelerator(LoggerHook):
    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
    
    def training_start(self, trainer: Trainer) -> None:
        if trainer.accelerator.is_main_process:
            assert hasattr(trainer.ctx, 'epochs')
            self.num_batches = len(trainer.data_loader)
            self.epochs = trainer.ctx.epochs
            self.logger.info(
                f"Training started with the following hyper parameters:\n"
                f"\tEpochs: {self.epochs}\n"
                f"\tTotal batch size {trainer.data_loader.total_batch_size}\n"
                f"\tLearning rate: {trainer.optimizer.defaults['lr']}"
            )
    
    def epoch_start(self, trainer: Trainer) -> None:
        if trainer.accelerator.is_main_process:
            super().epoch_start(trainer)

    def iteration_end(self, trainer: Trainer) -> None:
        step_loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()
        if trainer.accelerator.is_main_process:
            self.losses_per_batch.append(step_loss)
            idx = trainer.ctx.batch_idx + 1
            num_batches = len(trainer.data_loader)
            if idx % 10 == 0 or idx == num_batches:
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
                if hasattr(trainer.ctx, 'peek_results'):
                    for q, a in trainer.ctx.peek_results:
                        self.logger.info(f'Q:{q} A:{a}')

    def epoch_end(self, trainer: Trainer) -> None:
        if trainer.accelerator.is_main_process:
            super().epoch_end(trainer)
            