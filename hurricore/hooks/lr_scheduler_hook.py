from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import assign_args


class LRSchedulerHook(HurricoreHook):
    def __init__(
        self,
        trainer: HurricoreTrainer,
        lr_schedulers: list[LRScheduler] = None,
        modes: list[str] = None,
    ) -> None:
        super().__init__(trainer)
        self._validate_inputs(lr_schedulers, modes)
        self.modes = modes
        assign_args(self.init_context, locals())
    

    def __post_init__(self) -> None:
        self.lr_schedulers: list[LRScheduler] = []
        for lr_scheduler in self.init_context.lr_schedulers:
            self.trainer.accelerator.register_for_checkpointing(lr_scheduler)
            self.lr_schedulers.append(lr_scheduler)


    def _validate_inputs(self, lr_schedulers: list[LRScheduler], modes: list[str]) -> None:
        assert len(lr_schedulers) == len(modes), 'Length of lr_schedulers and modes should be the same.'
        for mode in modes:
            assert mode in ['per_epoch', 'per_step'], 'Invalid mode.'
        for lr_scheduler in lr_schedulers:
            assert lr_scheduler is not None, 'Invalid learning rate scheduler.'
            

    def execute_on_epoch_end(self) -> None:
        self._step_schedulers('per_epoch')


    def execute_on_step_end(self) -> None:
        if not self.trainer.accelerator.sync_gradients:
            return
        self._step_schedulers('per_step')


    def _step_schedulers(self, mode: str) -> None:
        for lr_scheduler, scheduler_mode in zip(self.lr_schedulers, self.modes):
            if scheduler_mode == mode:
                lr_scheduler.step()
