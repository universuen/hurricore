from pathlib import Path

from accelerate.utils import set_seed

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer


class CKPTHook(HookBase):
    def __init__(
        self, 
        folder_path: Path = None,
        interval: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.is_available = (folder_path is not None and folder_path.is_dir())
        if not self.is_available:
            return
        self.folder_path = folder_path
        self.interval = interval
        self.memory = {
            'dataloader': None,
            'batch_idx': 0,
        }
        self.seed = seed
        
    
    def on_training_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        
        trainer.accelerator.register_for_checkpointing(trainer.ctx)
        
        trainer.accelerator.dataloader_config.use_seedable_sampler = True
        set_seed(self.seed)
        trainer.data_loader._is_accelerate_prepared = False
        trainer.data_loader = trainer.accelerator.prepare(trainer.data_loader)
        
        ckpt_dirs = [d for d in self.folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_step_')]
        if len(ckpt_dirs) == 0:
            return 
        
        steps = [int(d.name.split('_')[-1]) for d in ckpt_dirs]
        latest_step = max(steps)
        latest_ckpt_dir = self.folder_path / f'ckpt_step_{latest_step}'

        trainer.accelerator.load_state(latest_ckpt_dir)
        
        trainer.ctx.epoch -= 1
        self.memory['dataloader'] = trainer.data_loader
        trainer.data_loader = trainer.accelerator.skip_first_batches(
            dataloader=trainer.data_loader, 
            num_batches=trainer.ctx.batch_idx,
        )
        self.memory['batch_idx'] = trainer.ctx.batch_idx
        if hasattr(trainer, 'logger') and trainer.accelerator.is_main_process:
            trainer.logger.info(f'Resumed training from checkpoint: {latest_ckpt_dir}')
    
    
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        step = trainer.ctx.global_step
        if step % self.interval == 0:
            current_batch_idx = trainer.ctx.batch_idx
            trainer.ctx.batch_idx = self.memory['batch_idx'] + current_batch_idx
            ckpt_path = self.folder_path / f'ckpt_step_{step}'
            trainer.accelerator.save_state(ckpt_path, safe_serialization=False)
            trainer.ctx.batch_idx = current_batch_idx
            
            if hasattr(trainer, 'logger') and trainer.accelerator.is_main_process:
                trainer.logger.info(f'Saved checkpoint at: {ckpt_path}')


    def on_epoch_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        self.memory['batch_idx'] = 0
        if self.memory['dataloader'] is not None:
            trainer.data_loader = self.memory['dataloader']
            self.memory['dataloader'] = None