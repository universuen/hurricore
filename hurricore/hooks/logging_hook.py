from __future__ import annotations

import sys
import time
import logging
from threading import Thread
from collections import deque
from pathlib import Path

import torch
import accelerate
from accelerate import Accelerator

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import (
    Logger,
    Dummy,
    ConfigBase,
    env,
    auto_name,
    get_list_mean,
    get_params_details_table,
    get_device_utilization,
    get_device_memory_info,
    get_trainable_layer_names,
)


class _DeviceMonitor:
    def __init__(
            self, 
            accelerator: Accelerator,
            interval: float = 1,
        ) -> None:
        self.accelerator = accelerator
        self.interval = interval
        self.data_queue = deque(maxlen=3)
        self.data_collecting_thread: Thread = None
    

    def run(self) -> None:
        self.data_collecting_thread = Thread(
            target=self._collect_data_in_background,
            daemon=True,  # set daemon to True to avoid blocking the main thread
        )
        self.data_collecting_thread.start()


    def _collect_data_in_background(self):
        while True:
            utilization = get_device_utilization()
            free_memory, total_memory = get_device_memory_info()
            used_memory = total_memory - free_memory
            self.data_queue.append((utilization, used_memory, total_memory))
            time.sleep(self.interval)


    def get_states(self) -> tuple[float, float, float]:
        # Compute means for utilization, used_memory, and total_memory in a single pass
        utilization, used_memory, total_memory = map(get_list_mean, zip(*self.data_queue))
        # Create a single tensor with all values to batch operations
        states = torch.tensor(
            [[utilization, used_memory, total_memory]],
            dtype=torch.float32,
            device=self.accelerator.device
        )
        # Gather tensors in one go and compute results
        gathered_states = self.accelerator.gather(states)
        # Ensure gathered_states is summed along dimension 0
        gathered_sums = gathered_states.sum(dim=0)
        # Extract scalar values correctly
        utilization_mean = gathered_sums[0].item() / self.accelerator.num_processes
        used_memory_total = gathered_sums[1].item()
        total_memory_total = gathered_sums[2].item()
        return utilization_mean, used_memory_total, total_memory_total


class _TimeEstimator:
    def __init__(
        self,
        num_epochs: int,
        num_steps_per_epoch: int,
    ):
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.start_time: float = time.time()
        self.num_passed_iterations = 0


    def reset(self):
        self.start_time = time.time()
        self.num_passed_iterations = 0
    

    def step(self):
        self.num_passed_iterations += 1


    def get_estimated_remaining_time(
        self, 
        epoch: int,
        batch_idx: int,
    ):
        elapsed_time = time.time() - self.start_time
        remaining_epochs = self.num_epochs - epoch - 1
        remaining_iterations_in_epoch = self.num_steps_per_epoch - batch_idx - 1
        num_iterations = self.num_steps_per_epoch
        remaining_global_interations = remaining_epochs * num_iterations + remaining_iterations_in_epoch
        avg_time_per_iteration = elapsed_time / (self.num_passed_iterations + 1e-6)
        remaining_time = remaining_global_interations * avg_time_per_iteration
        days, remainder = divmod(remaining_time, 86400) 
        hours, remainder = divmod(remainder, 3600) 
        minutes, _ = divmod(remainder, 60) 
        if days > 0:  # days > 0
            formatted_remaining_time = f"{int(days)}d {int(hours):02d}h {int(minutes):02d}m"
        elif hours > 0:  # days == 0 and hours > 0
            formatted_remaining_time = f"{int(hours)}h {int(minutes):02d}m"
        else:  # days == 0 and hours == 0
            formatted_remaining_time = f"{int(minutes):02d}m"
        return formatted_remaining_time


class _LossManager:
    def __init__(
        self,
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.losses = []


    def reset(self):
        self.losses = []
    

    def update(self, loss: torch.Tensor):
        step_loss = self.accelerator.gather(loss).mean().item()
        self.losses.append(step_loss)
    

    @property
    def average_loss(self):
        return get_list_mean(self.losses)
    
    
    @property
    def latest_loss(self):
        return self.losses[-1]


class LoggingHook(HurricoreHook):
    def __init__(
        self, 
        trainer: HurricoreTrainer,
        experiment_name: str,
        interval: int = 1, 
        level=logging.INFO,
        folder_path: Path = None,
        log_trainable_params: bool = False,
    ) -> None:
        super().__init__(
            trainer=trainer,
            experiment_name=experiment_name,
            interval=interval,
            level=level,
            folder_path=folder_path,
            log_trainable_params=log_trainable_params,
        )
        assert interval > 0, 'Log interval must be greater than 0.'
        assert isinstance(folder_path, Path), f'Invalid logging folder path. Expected Path, but got {type(folder_path)}.'
        self.interval = interval
        self.extra_log_bar_data = dict()
    

    def execute_on_trainer_init_end(self) -> None:
        logger = Logger(
            name=self.init_context.experiment_name,
            level=self.init_context.level,
            logs_dir=self.init_context.folder_path,
        )
        self.logger = logger if self.trainer.accelerator.is_main_process else Dummy()
        self.all_process_logger = logger
        # Attach logger to Environment for global access
        env.logger = self.logger
        env.all_process_logger = self.all_process_logger
        # Wrap trainer run method with exception logging
        self._wrap_trainer_run_method_with_exception_logging()
        self.logger.info(f'Loaded config from {env.config_path}')
        # Setup helpers
        self.device_monitor = _DeviceMonitor(
            accelerator=self.trainer.accelerator
        )
        self.time_estimator = _TimeEstimator(
            num_epochs=self.trainer.context.num_epochs,
            num_steps_per_epoch=self.trainer.context.num_steps_per_epoch,
        )
        self.loss_manager = _LossManager(
            accelerator=self.trainer.accelerator
        )

    
    def _wrap_trainer_run_method_with_exception_logging(self) -> None:
        original_run_method = self.trainer.run
        def wrapped_run_method(*args, **kwargs):
            try:
                original_run_method(*args, **kwargs)
            except Exception as e:
                self.all_process_logger.error(f'Exception occurred on device {self.trainer.accelerator.process_index}')
                self.all_process_logger.exception(e)
                raise 
        self.trainer.run = wrapped_run_method
    
    
    def execute_on_training_start(self) -> None:
        self.logger.info("Components versions:")
        self.logger.info(f"\t`python`: {sys.version}")
        self.logger.info(f'\t`torch`: {torch.__version__}')
        self.logger.info(f"\t`accelerate`: {accelerate.__version__}")
        for subclass in ConfigBase.__subclasses__():
            self.logger.info(f"{subclass.__name__}\n{subclass()}")
        self.logger.info(f'Running with {self.trainer.accelerator.num_processes} processes')
        self.logger.info(f'Trainer & Hooks:\n{self.trainer}')
        models = self.trainer.originals.models
        for name, model in zip(auto_name(models), models):
            self.logger.info(f'{name} structure:\n{model}')
            if self.init_context.log_trainable_params:
                self.logger.info(f"{name} trainable layers:\n\t" + "\n\t".join(get_trainable_layer_names(model)))
        params_table = get_params_details_table(*models)
        self.logger.info(f'Parameters details:\n{params_table}')
        self.trainer.accelerator.wait_for_everyone()
        self.time_estimator.reset()
        self.device_monitor.run()
        self.logger.info('Training started')
    
    
    def execute_on_epoch_start(self) -> None:
        self.loss_manager.reset()
        self.logger.info(f'Epoch {self.trainer.context.epoch + 1} started')
    
        
    def execute_on_step_end(self) -> None:
        self.time_estimator.step()
        if (self.trainer.context.global_step + 1) % self.interval == 0:
            self._log_states_bar()
    
      
    def execute_on_epoch_end(self) -> None: 
        avg_loss = self.loss_manager.average_loss
        self.logger.info(f'Epoch {self.trainer.context.epoch + 1} finished with average loss: {avg_loss: .5f}')
    
    
    def execute_on_training_end(self) -> None:
        self.logger.info('Training finished')

    
    def _log_states_bar(self):
        self.loss_manager.update(self.trainer.context.step_loss)
        remaining_time = self.time_estimator.get_estimated_remaining_time(
            epoch=self.trainer.context.epoch,
            batch_idx=self.trainer.context.batch_idx,
        )
        device_utilization, used_memory, total_memory = self.device_monitor.get_states()

        idx = self.trainer.context.batch_idx + 1
        num_epochs = self.trainer.context.num_epochs
        num_steps_per_epoch = self.trainer.context.num_steps_per_epoch
        epoch = self.trainer.context.epoch + 1
        progress = (self.trainer.context.global_step + 1) / (num_epochs * num_steps_per_epoch)
        
        # LR patch
        lr_status = f"{self.trainer.optimizer.param_groups[0]['lr']:.2e}"
        if len(self.trainer.optimizers) > 1:
            lr_status = ""
            for optimizer_name, optimizer in zip(auto_name(self.trainer.optimizers), self.trainer.optimizers):
                lr_status += f"[{optimizer_name}: {optimizer.param_groups[0]['lr']:.2e}]"

        states = (
            f"Epoch: {epoch}/{num_epochs} | "
            f"Batch: {idx}/{num_steps_per_epoch} | "
            f"Loss: {self.loss_manager.latest_loss:.5f} | "
            f"LR: {lr_status} | "
            f"Prog.: {progress:.2%} | "
            f"T.L.: {remaining_time} | "
            f"Util.: {device_utilization:.1f}% "
            f"{used_memory / 1024 ** 3:.1f}/{total_memory / 1024 ** 3:.1f} GB"
        )
        for key, value in self.extra_log_bar_data.items():
            states += f' | {key}: {value}'
        self.logger.info(states)
