from pathlib import Path

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer_base import TrainerBase


class CheckpointHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        folder_path: Path = None,
        interval: int = 100,
    ) -> None:
        super().__init__(trainer)
        self.is_available = (folder_path is not None and folder_path.is_dir())
        if not self.is_available:
            return
        self.folder_path = folder_path
        self.interval = interval
        
    
    def on_training_start(self) -> None:
        if not self.is_available:
            return
        self.trainer.accelerator.register_for_checkpointing(self.trainer.ctx)
        if not self.trainer.accelerator.dataloader_config.use_seedable_sampler:
            raise ValueError(
                """
                For deterministic reproducibility, 
                CheckpointHook requires `use_seedable_sampler=True` in `DataLoaderConfiguration` of `Accelerator`.
                Try: `Accelerator(dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True))`
                """
            )
        
        # TODO: Remove this when the issue is fixed in Accelerate `prepare_dataloader()`#####################
        if hasattr(self.trainer.data_loader.batch_sampler, 'batch_sampler'):
            self.trainer.data_loader.batch_sampler.batch_sampler.sampler = self.trainer.data_loader.batch_sampler.sampler
        #####################################################################################################
        
        ckpt_dirs = [d for d in self.folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_step_')]
        if len(ckpt_dirs) == 0:
            return 
        
        steps = [int(d.name.split('_')[-1]) for d in ckpt_dirs]
        latest_step = max(steps)
        latest_ckpt_dir = self.folder_path / f'ckpt_step_{latest_step}'

        self.trainer.accelerator.load_state(latest_ckpt_dir)
        
        self.trainer.data_loader.set_epoch(self.trainer.ctx.epoch - 1)
        self.trainer.data_loader.skip_batches = self.trainer.ctx.batch_idx
        
        self.trainer.ctx.epoch -= 1
        # self.trainer.data_loader.skip_batches = self.trainer.ctx.batch_idx
        if hasattr(self.trainer, 'logger') and self.trainer.accelerator.is_main_process:
            self.trainer.logger.info(f'Resumed training from checkpoint: {latest_ckpt_dir}')
    
    
    def on_step_end(self) -> None:
        if not self.is_available:
            return
        step = self.trainer.ctx.global_step
        if step % self.interval == 0:
            current_batch_idx = self.trainer.ctx.batch_idx
            self.trainer.ctx.batch_idx = self.trainer.data_loader.skip_batches + current_batch_idx
            ckpt_path = self.folder_path / f'ckpt_step_{step}'
            self.trainer.accelerator.save_state(ckpt_path, safe_serialization=False)
            self.trainer.ctx.batch_idx = current_batch_idx
            
            if hasattr(self.trainer, 'logger') and self.trainer.accelerator.is_main_process:
                self.trainer.logger.info(f'Saved checkpoint at: {ckpt_path}')


    def on_epoch_end(self) -> None:
        if not self.is_available:
            return
        self.trainer.data_loader.skip_batches = 0