from logging import Logger
import time

from torch.cuda import memory_cached

from src.hooks.hook import Hook
from src.trainers.trainer import Trainer
from src.utils import get_list_mean


class LoggerHook(Hook):
    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger
    
    def training_start(self, trainer: Trainer) -> None:
        assert hasattr(trainer.ctx, 'epochs')
        self.num_batches = len(trainer.data_loader)
        self.epochs = trainer.ctx.epochs
        self.logger.info(
            f"\tTraining started with the following hyper parameters:\n"
            f"\tEpochs: {self.epochs}\n"
            f"\tBatch size: {trainer.data_loader.batch_size}\n"
            f"\tLearning rate: {trainer.optimizer.defaults['lr']}"
        )
    
    def epoch_start(self, trainer: Trainer) -> None:
        assert hasattr(trainer.ctx, 'epoch')
        self.losses_per_batch = []
        self.epoch = trainer.ctx.epoch
        self.logger.info(f'Epoch {self.epoch} started')
        self.start_time = time.time()
    
    def iteration_end(self, trainer: Trainer) -> None:
        assert hasattr(trainer.ctx, 'step_loss')
        step_loss = trainer.ctx.step_loss
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
    
    def epoch_end(self, trainer: Trainer) -> None:
        avg_loss = get_list_mean(self.losses_per_batch)
        self.logger.info(f'Epoch {self.epoch} finished with average loss: {avg_loss}')

