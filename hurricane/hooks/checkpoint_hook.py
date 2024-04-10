from pathlib import Path
from time import sleep

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
        # re-prepare dataloader with seedable sampler if 
        conditions = [
            any(isinstance(dl.sampler, RandomSampler) for dl in trainer.originals.data_loaders),
            trainer.accelerator.dataloader_config.use_seedable_sampler is False,
        ]
        if all(conditions) is True:
            torch.manual_seed(seed)
            LoggerHook.msg_queue.append(('info', 'To ensure reproducibility, dataloaders are reprepared with seedable sampler.'))
            trainer.accelerator.dataloader_config.use_seedable_sampler = True
            trainer.data_loaders = [
                trainer.accelerator.prepare(dl) 
                for dl in trainer.originals.data_loaders
            ]
        
        # register trainer context for checkpointing
        trainer.accelerator.register_for_checkpointing(trainer.ctx)
        # setup self
        self.folder_path = folder_path
        self.interval = interval
    
    
    def on_training_start(self) -> None:
        # check available checkpoint
        ckpt_dirs = [d for d in self.folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_step_')]
        if len(ckpt_dirs) == 0:
            return 
        # load latest checkpoint
        steps = [int(d.name.split('_')[-1]) for d in ckpt_dirs]
        latest_step = max(steps)
        latest_ckpt_dir = self.folder_path / f'ckpt_step_{latest_step}'
        self.trainer.accelerator.load_state(latest_ckpt_dir)
        # should step into the next batch
        self.trainer.ctx.batches_idx += 1
        # recover dataloaders states
        for dl in self.trainer.data_loaders:
            dl.set_epoch(self.trainer.ctx.epoch)
            dl.skip_batches = self.trainer.ctx.batches_idx
        # recover hooks
        for hook in self.trainer.hooks:
            if hasattr(hook, 'recover_from_checkpoint'):
                hook.recover_from_checkpoint()
        self.trainer.accelerator.wait_for_everyone()
        # log
        LoggerHook.msg_queue.append(('info', f'Resumed training from checkpoint: {latest_ckpt_dir}'))
    
    
    def on_step_end(self) -> None:
        step = self.trainer.ctx.global_step + 1
        if step % self.interval == 0:
            self._save_checkpoint()


    def on_epoch_end(self) -> None:
        for dl in self.trainer.data_loaders:
            dl.skip_batches = 0
        if self.trainer.ctx.epoch + 1 == self.trainer.epochs:
            '''
            This cannot be put in `on_training_end` because LoggerHook will
            shut down its message queue before `on_training_end` is called.
            '''
            self._save_checkpoint()
            sleep(1)

    def _save_checkpoint(self) -> None:
        step = self.trainer.ctx.global_step + 1
        ckpt_path = self.folder_path / f'ckpt_step_{step}'
        self.trainer.accelerator.save_state(ckpt_path, safe_serialization=False)
        LoggerHook.msg_queue.append(('info', f'Saved checkpoint at: {ckpt_path}'))
