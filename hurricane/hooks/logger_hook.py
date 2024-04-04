import time
from logging import Logger
from threading import Thread

from torch.cuda import memory_reserved

from hurricane.hooks import HookBase
from hurricane.trainers import TrainerBase
from hurricane.utils import (
    DummyObject,
    auto_name,
    get_list_mean,
    get_total_parameters,
    get_trainable_parameters,
)


class LoggerHook(HookBase):
    
    msg_queue = []
    
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
        self.interval = interval
        # logger is only for main process
        self.logger = logger if trainer.accelerator.is_main_process else DummyObject()
        self._activate_msg_queue()
    
    
    def on_training_start(self) -> None:
        self.logger.info('Training started')
        self.logger.info(f'Trainer:\n{self.trainer}')
        models = self.trainer.originals.models
        for name, model in zip(auto_name(models), models):
            self.logger.info(f'{name} structure:\n{model}')
            self.logger.info(f'{name} total parameters: {get_total_parameters(model)}')
            self.logger.info(f'{name} trainable parameters: {get_trainable_parameters(model)}')
    
    
    def on_epoch_start(self) -> None:
        self.step = 0
        self.step_losses = []
        self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} started')
        self.start_time = time.time()
    
        
    def on_step_end(self) -> None:
        self.step += 1
        if (self.trainer.ctx.global_step + 1) % self.interval == 0:
            self._collect_step_loss()
            self._log_states()
      
      
    def on_epoch_end(self) -> None: 
        self._log_states()
        avg_loss = get_list_mean(self.step_losses)
        self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} finished with average loss: {avg_loss: .5f}')


    def _collect_step_loss(self):
        step_loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
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
    
    def _activate_msg_queue(self):
        def listen_and_process(self):
            while True:
                if len(self.msg_queue) > 0:
                    try:
                        method, msg = self.msg_queue.pop(0)
                        getattr(self.logger, method)(msg)
                    except Exception as e:
                        self.logger.error(f'Error in LoggerHook: {e}')
                else:
                    time.sleep(0.01)
        Thread(target=listen_and_process, args=(self, )).start()
    