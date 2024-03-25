from torch.optim.lr_scheduler import LRScheduler

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer_base import TrainerBase


class LRSchedulerHook(HookBase):
    def __init__(
        self,
        trainer: TrainerBase,
        lr_scheduler: LRScheduler = None,
        mode: str = 'per_epoch',
    ) -> None:
        super().__init__(trainer)
        assert mode in ('per_epoch', 'per_step')
        self.is_available = (lr_scheduler is not None)
        if not self.is_available:
            return
        self.lr_scheduler = lr_scheduler
        self.mode = mode
        self.lr_scheduler = trainer.accelerator.prepare(self.lr_scheduler)
        trainer.accelerator.step_scheduler_with_optimizer = (self.mode == 'per_step')
        
    def on_epoch_end(self) -> None:
        if not self.is_available or self.mode != 'per_epoch':
            return
        self.lr_scheduler.step()
    
    def on_step_end(self) -> None:
        if not self.is_available or self.mode != 'per_step':
            return
        self.lr_scheduler.step()
