import time
from logging import Logger
from threading import Thread

from torch.cuda import mem_get_info, utilization

from hurricane.hooks import Hook
from hurricane.trainers import Trainer
from hurricane.utils import (
    DummyObject,
    ConfigBase,
    auto_name,
    get_list_mean,
    get_params_details_table,
)


class LoggerHook(Hook):
    
    msg_queue = []
    
    def __init__(
        self, 
        trainer: Trainer,
        logger: Logger,
        interval: int = 1, 
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert interval > 0, 'Log interval must be greater than 0.'
        assert logger is not None, 'Invalid logger.'
        # setup self
        self.interval = interval
        # logger is only for main process
        self.logger = logger if trainer.accelerator.is_main_process else DummyObject()
        self._activate_msg_queue()
        self._wrap_trainer_run_method_with_exception_logging()
    
    
    def _wrap_trainer_run_method_with_exception_logging(self) -> None:
        original_run_method = self.trainer.run

        def wrapped_run_method(*args, **kwargs):
            try:
                original_run_method(*args, **kwargs)
            except Exception as e:
                self.logger.exception(e)

        self.trainer.run = wrapped_run_method
    
    
    def on_training_start(self) -> None:
        for subclass in ConfigBase.__subclasses__():
            self.logger.info(subclass())
        self.logger.info('Training started')
        self.logger.info(f'Trainer:\n{self.trainer}')
        models = self.trainer.originals.models
        for name, model in zip(auto_name(models), models):
            self.logger.info(f'{name} structure:\n{model}')
        params_table = get_params_details_table(*models)
        self.logger.info(f'Parameters details:\n{params_table}')
        
        self.start_time = time.time()
        self.num_passed_iterations = 0
    
    
    def on_epoch_start(self) -> None:
        self.step_losses = []
        self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} started')
    
        
    def on_step_end(self) -> None:
        self.num_passed_iterations += 1
        if (self.trainer.ctx.global_step + 1) % self.interval == 0:
            self._collect_step_loss()
            self._log_states()
      
      
    def on_epoch_end(self) -> None: 
        avg_loss = get_list_mean(self.step_losses)
        self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} finished with average loss: {avg_loss: .5f}')
    
    
    def _collect_step_loss(self):
        step_loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
        self.step_losses.append(step_loss)
    
    
    def _get_remaining_time(self):
        elapsed_time = time.time() - self.start_time
        remaining_epochs = self.trainer.num_epochs - self.trainer.ctx.epoch - 1
        remaining_iterations_in_epoch = self.trainer.ctx.num_steps_per_epoch - self.trainer.ctx.batches_idx - 1
        num_iterations = self.trainer.ctx.num_steps_per_epoch
        remaining_global_interations = remaining_epochs * num_iterations + remaining_iterations_in_epoch
        avg_time_per_iteration = elapsed_time / (self.num_passed_iterations + 1e-6)
        remaining_time = remaining_global_interations * avg_time_per_iteration
        days, remainder = divmod(remaining_time, 86400) 
        hours, remainder = divmod(remainder, 3600) 
        minutes, seconds = divmod(remainder, 60) 
        formatted_remaining_time = f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return formatted_remaining_time
    
    
    def _log_states(self):
        if len(self.step_losses) == 0:
            return
        idx = self.trainer.ctx.batches_idx + 1
        num_epochs = self.trainer.num_epochs
        num_steps_per_epoch = self.trainer.ctx.num_steps_per_epoch
        epoch = self.trainer.ctx.epoch + 1
        progress = (self.trainer.ctx.global_step + 1) / (num_epochs * num_steps_per_epoch)
        remaining_time = self._get_remaining_time()
        
        free_memory, total_memory = mem_get_info()
        used_memory = total_memory - free_memory
        
        self.logger.info(
            f"Epoch: {epoch}/{self.trainer.num_epochs} | "
            f"Step: {idx}/{num_steps_per_epoch} | "
            f"Loss: {self.step_losses[-1]:.5f} | "
            f"Progress: {progress:.2%} | "
            f"Time left: {remaining_time} | "
            f"GPU usage: {utilization()}% w. {used_memory / 1024 ** 3:.2f} GB"
        )
    
    
    def _activate_msg_queue(self):
        def listen_and_process(self):
            while True:
                if len(self.msg_queue) > 0:
                    try:
                        method, msg = self.msg_queue.pop(0)
                        getattr(self.logger, method)(msg)
                    except Exception as e:
                        self.logger.exception(e)
                else:
                    time.sleep(0.01)
        Thread(target=listen_and_process, args=(self, ), daemon=True).start()
    