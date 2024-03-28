from pathlib import Path

import torch
from torch.utils.data import RandomSampler

from hurricane.hooks import HookBase, LoggerHook
from hurricane.trainers import TrainerBase


class CheckpointHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        folder_path: Path = None,
        interval: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert interval > 0, 'Checkpoint interval must be greater than 0.'
        assert folder_path is not None and folder_path.is_dir(), 'Invalid checkpoint folder path.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        self.msg_queue = []
        if isinstance(trainer.originals.data_loader.sampler, RandomSampler):
            # reprepare dataloader with seedable sampler
            torch.manual_seed(seed)
            if trainer.accelerator.dataloader_config.use_seedable_sampler is False:
                self.msg_queue.append(
                    (
                        'warning', 
                        'To ensure reproducibility, the dataloader is reprepared with seedable sampler.\n'
                        'To avoid this, set `accelerator.dataloader_config.use_seedable_sampler=True`.'
                    )
                )
            trainer.accelerator.dataloader_config.use_seedable_sampler = True
            trainer.data_loader = trainer.accelerator.prepare(trainer.originals.data_loader)
            
            # TODO: Remove this when the issue is fixed in Accelerate `prepare_dataloader()`#####################
            try:
                if hasattr(self.trainer.data_loader.batch_sampler, 'batch_sampler'):
                    self.trainer.data_loader.batch_sampler.batch_sampler.sampler = self.trainer.data_loader.batch_sampler.sampler
            except AttributeError:
                self.msg_queue.append(
                    (
                        'error', 
                        'Failed to fix the issue in Accelerate `prepare_dataloader()`.'
                    )
                )
            ##################################################################################################### 
        
        trainer.accelerator.register_for_checkpointing(trainer.ctx)
        # setup self
        self.folder_path = folder_path
        self.interval = interval
        
    def on_training_start(self) -> None:
        # collect logger
        logger_hook = self.trainer.get_hook(LoggerHook)
        if logger_hook is not None:
            self.logger = logger_hook.logger
            # process message queue
            while len(self.msg_queue) > 0:
                msg_type, msg = self.msg_queue.pop(0)
                getattr(self.logger, msg_type)(msg)
            del self.msg_queue
        # check available checkpoint
        ckpt_dirs = [d for d in self.folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_step_')]
        if len(ckpt_dirs) == 0:
            return 
        # load latest checkpoint
        steps = [int(d.name.split('_')[-1]) for d in ckpt_dirs]
        latest_step = max(steps)
        latest_ckpt_dir = self.folder_path / f'ckpt_step_{latest_step}'
        self.trainer.accelerator.load_state(latest_ckpt_dir)
        # recover dataloader state
        self.trainer.data_loader.set_epoch(self.trainer.ctx.epoch - 1)
        self.trainer.data_loader.skip_batches = self.trainer.ctx.batch_idx
        # recover context state
        self.trainer.ctx.epoch -= 1
        # log
        if hasattr(self, 'logger') and self.trainer.accelerator.is_main_process:
            self.logger.info(f'Resumed training from checkpoint: {latest_ckpt_dir}')
    
    
    def on_step_end(self) -> None:
        step = self.trainer.ctx.global_step
        conditions = [
            step % self.interval == 0,
            self.trainer.ctx.epoch == self.trainer.epochs,
            self.trainer.ctx.batch_idx == len(self.trainer.data_loader),
        ]
        if conditions[0] or (conditions[1] and conditions[2]):
            current_batch_idx = self.trainer.ctx.batch_idx
            self.trainer.ctx.batch_idx = self.trainer.data_loader.skip_batches + current_batch_idx
            ckpt_path = self.folder_path / f'ckpt_step_{step}'
            self.trainer.accelerator.save_state(ckpt_path, safe_serialization=False)
            self.trainer.ctx.batch_idx = current_batch_idx
            
            if hasattr(self, 'logger') and self.trainer.accelerator.is_main_process:
                self.logger.info(f'Saved checkpoint at: {ckpt_path}')

    def on_epoch_end(self) -> None:
        if not self.is_available:
            return
        self.trainer.data_loader.skip_batches = 0
