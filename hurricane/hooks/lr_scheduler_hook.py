from torch.optim.lr_scheduler import LRScheduler

from hurricane.hooks import HookBase, LoggerHook, TensorBoardHook
from hurricane.trainers import TrainerBase


class LRSchedulerHook(HookBase):
    def __init__(
        self,
        trainer: TrainerBase,
        lr_scheduler: LRScheduler = None,
        mode: str = 'per_epoch',
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert mode in ['per_epoch', 'per_step'], 'Invalid mode.'
        assert lr_scheduler is not None, 'Invalid learning rate scheduler.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        # setup trainer
        trainer.originals.lr_scheduler = lr_scheduler
        trainer.accelerator.step_scheduler_with_optimizer = (mode == 'per_step')
        self.msg_queue = [
            (
                'info',
                f'Set `accelerator.step_scheduler_with_optimizer` to {mode == "per_step"}',
            )
        ]
        # setup self
        self.mode = mode
        self.lr_scheduler = trainer.accelerator.prepare(lr_scheduler)
    
    def on_training_start(self) -> None:
        # collect logger
        logger_hook = self.trainer.get_hook(LoggerHook)
        if logger_hook is not None:
            self.logger = logger_hook.logger
            self.log_interval = logger_hook.interval
        # collect tensor board writer
        tb_hook = self.trainer.get_hook(TensorBoardHook)
        if tb_hook is not None:
            self.get_temp_tb_writer = tb_hook.get_temp_writer
            self.tb_interval = tb_hook.interval
            
    def on_epoch_end(self) -> None:
        if self.mode != 'per_epoch':
            return
        self.lr_scheduler.step()
        # log learning rate
        conditions = (
            hasattr(self, 'logger'),
            self.trainer.accelerator.is_main_process,
        )
        if all(conditions) is True:
            self.logger.info(f'Learning rate: {self.lr_scheduler.get_last_lr()}')
        # write learning rate to tensorboard
        conditions = (
            hasattr(self, 'get_temp_tb_writer'),
            self.trainer.accelerator.is_main_process,
        )
        if all(conditions) is True:
            tb_writer = self.get_temp_tb_writer()
            for idx, lr in enumerate(self.lr_scheduler.get_last_lr()):
                tb_writer.add_scalar(
                    tag = f'Learning Rate/{idx}',
                    scalar_value = lr,
                    global_step = self.trainer.ctx.global_step,
                )
            tb_writer.close()
    
    def on_step_end(self) -> None:
        if self.mode != 'per_step':
            return
        self.lr_scheduler.step()
        # log the learning rate
        conditions = (
            hasattr(self, 'logger'),
            hasattr(self, 'log_interval'),
            self.trainer.accelerator.is_main_process,
            self.trainer.ctx.global_step % self.log_interval == 0,
        )
        if all(conditions) is True:
            self.logger.info(f'Learning rate: {self.lr_scheduler.get_last_lr()}')
        # log the learning rate to tensorboard
        conditions = (
            hasattr(self, 'get_temp_tb_writer'),
            hasattr(self, 'tb_interval'),
            self.trainer.accelerator.is_main_process,
            self.trainer.ctx.global_step % self.log_interval == 0,
        )
        if all(conditions) is True:
            tb_writer = self.get_temp_tb_writer()
            for idx, lr in enumerate(self.lr_scheduler.get_last_lr()):
                tb_writer.add_scalar(
                    tag = f'Learning Rate/{idx}',
                    scalar_value = lr,
                    global_step = self.trainer.ctx.global_step,
                )
            tb_writer.close()
