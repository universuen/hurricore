from torch.optim.lr_scheduler import LRScheduler

from hurricane.hooks import HookBase, LoggerHook, TensorBoardHook
from hurricane.trainers import TrainerBase
from hurricane.utils import auto_name
from hurricane.context import Context


class LRSchedulerHook(HookBase):
    def __init__(
        self,
        trainer: TrainerBase,
        lr_schedulers: list[LRScheduler] = None,
        mode: str = 'per_epoch',
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert mode in ['per_epoch', 'per_step'], 'Invalid mode.'
        for lr_scheduler in lr_schedulers:
            assert lr_scheduler is not None, 'Invalid learning rate scheduler.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        # setup trainer
        self.originals = Context(
            lr_schedulers=lr_schedulers,
        )
        trainer.accelerator.step_scheduler_with_optimizer = (mode == 'per_step')
        self.msg_queue = []
        if self.trainer.accelerator.is_main_process:
            self.msg_queue.append(
                (
                    'info',
                    f'Set `accelerator.step_scheduler_with_optimizer` to {mode == "per_step"}',
                )
            )
        # setup self
        self.mode = mode
        self.lr_schedulers = [
            trainer.accelerator.prepare(lr_scheduler) 
            for lr_scheduler in lr_schedulers
        ]
    
    def on_training_start(self) -> None:
        # collect logger
        logger_hook = self.trainer.get_hook(LoggerHook)
        conditions = (
            logger_hook is not None,
            self.trainer.accelerator.is_main_process,
        )
        if all(conditions) is True:
            self.logger = logger_hook.logger
            self.log_interval = logger_hook.interval
            # process message queue
            while len(self.msg_queue) > 0:
                msg_type, msg = self.msg_queue.pop(0)
                getattr(self.logger, msg_type)(msg)
            del self.msg_queue
        # collect tensor board writer
        tb_hook = self.trainer.get_hook(TensorBoardHook)
        conditions = (
            tb_hook is not None,
            self.trainer.accelerator.is_main_process,
        )
        if all(conditions) is True:
            self.writer = tb_hook.writer
            self.tb_interval = tb_hook.interval
            
    def on_epoch_end(self) -> None:
        if self.mode != 'per_epoch':
            return
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        # log learning rate
        if hasattr(self, 'logger'):
            lr_schedulers = self.originals.lr_schedulers
            optimizers = [lr_scheduler.optimizer for lr_scheduler in lr_schedulers]
            for name, lr_scheduler in zip(auto_name(optimizers), lr_schedulers):
                msg = f'{name} LR: {'|'.join([f"{lr:.7f}" for lr in lr_scheduler.get_last_lr()])}'
                self.logger.info(msg)
        # write learning rate to tensorboard
        if hasattr(self, 'writer'):
            lr_schedulers = self.originals.lr_schedulers
            optimizers = [lr_scheduler.optimizer for lr_scheduler in lr_schedulers]
            for scheduler_name, lr_scheduler in zip(auto_name(optimizers), lr_schedulers):
                for idx, lr in enumerate(lr_scheduler.get_last_lr()):
                    self.writer.add_scalar(
                        tag = f'Learning Rate/{scheduler_name} group_{idx}',
                        scalar_value = lr,
                        global_step = self.trainer.ctx.global_step,
                    )
            self.writer.flush()
    
    def on_step_end(self) -> None:
        if self.mode != 'per_step':
            return
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        # log the learning rate
        conditions = (
            hasattr(self, 'logger'),
            hasattr(self, 'log_interval'),
        )
        if all(conditions) and (self.trainer.ctx.global_step + 1) % self.log_interval == 0:
            lr_schedulers = self.originals.lr_schedulers
            optimizers = [lr_scheduler.optimizer for lr_scheduler in lr_schedulers]
            for name, lr_scheduler in zip(auto_name(optimizers), lr_schedulers):
                msg = f'{name} LR: {'|'.join([f"{lr:.7f}" for lr in lr_scheduler.get_last_lr()])}'
                self.logger.info(msg)
        # log the learning rate to tensorboard
        conditions = (
            hasattr(self, 'writer'),
            hasattr(self, 'tb_interval'),
        )
        if all(conditions) and (self.trainer.ctx.global_step + 1) % self.tb_interval == 0:
            lr_schedulers = self.originals.lr_schedulers
            optimizers = [lr_scheduler.optimizer for lr_scheduler in lr_schedulers]
            for scheduler_name, lr_scheduler in zip(auto_name(optimizers), lr_schedulers):
                for idx, lr in enumerate(lr_scheduler.get_last_lr()):
                    self.writer.add_scalar(
                        tag = f'Learning Rate/{scheduler_name} group_{idx}',
                        scalar_value = lr,
                        global_step = self.trainer.ctx.global_step,
                    )
            self.writer.flush()
