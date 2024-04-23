from torch.optim.lr_scheduler import LRScheduler

from hurricane.hooks import Hook, LoggerHook, TensorBoardHook
from hurricane.trainers import Trainer
from hurricane.utils import Context, auto_name


class LRSchedulerHook(Hook):
    def __init__(
        self,
        trainer: Trainer,
        lr_schedulers: list[LRScheduler] = None,
        mode: str = 'per_epoch',
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert mode in ['per_epoch', 'per_step'], 'Invalid mode.'
        for lr_scheduler in lr_schedulers:
            assert lr_scheduler is not None, 'Invalid learning rate scheduler.'
        # setup trainer
        self.originals = Context(
            lr_schedulers=lr_schedulers,
        )
        trainer.accelerator.step_scheduler_with_optimizer = (mode == 'per_step')
        LoggerHook.msg_queue.append(('info', f'Set `accelerator.step_scheduler_with_optimizer` to {mode == "per_step"}'))
        # setup self
        self.mode = mode
        self.lr_schedulers = [
            trainer.accelerator.prepare(lr_scheduler) 
            for lr_scheduler in lr_schedulers
        ]
        self._lr_records = [lr_scheduler.get_last_lr() for lr_scheduler in self.lr_schedulers]
    
    
    def on_epoch_end(self) -> None:
        if self.mode != 'per_epoch':
            return
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        self._update_lr_records()

    
    def on_step_end(self) -> None:
        if self.mode != 'per_step':
            return
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        self._update_lr_records()
    
    
    def _update_lr_records(self):
        current_lrs = [lr_scheduler.get_last_lr() for lr_scheduler in self.lr_schedulers]
        if all([current == previous for current, previous in zip(current_lrs, self._lr_records)]):
            return
        self._lr_records = current_lrs
        lr_schedulers = self.originals.lr_schedulers
        optimizers = [lr_scheduler.optimizer for lr_scheduler in lr_schedulers]
        for name, lr_scheduler in zip(auto_name(optimizers), lr_schedulers):
            lr_string = '|'.join([f"{lr:.7f}" for lr in lr_scheduler.get_last_lr()])
            msg = f"{name} LR: {lr_string}"
            LoggerHook.msg_queue.append(('info', msg))
            for idx, lr in enumerate(lr_scheduler.get_last_lr()):
                TensorBoardHook.msg_queue.append(
                    (
                        'add_scalar',
                        {
                            'tag': f'Learning Rate/{name} group_{idx}',
                            'scalar_value': lr,
                            'global_step': self.trainer.ctx.global_step,
                        }
                    )
                )
