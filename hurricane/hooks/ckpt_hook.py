from pathlib import Path

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer
from hurricane.trainers.trainer_base import TrainerBase


class CKPTHook(HookBase):
    def __init__(
        self, 
        folder_path: Path = None,
    ) -> None:
        super().__init__()
        self.is_available = (folder_path is not None and folder_path.is_dir())
        self.folder_path = folder_path
        self.cnt = 0
    
    def on_training_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        
        trainer.accelerator.register_for_checkpointing(trainer.ctx)
        
        ckpt_dirs = [d for d in self.folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_epoch_')]
        if not ckpt_dirs:
            return 
        
        steps = [int(d.name.split('_')[-1]) for d in ckpt_dirs]
        latest_step = max(steps)
        latest_ckpt_dir = self.folder_path / f'ckpt_epoch_{latest_step}'

        trainer.accelerator.load_state(latest_ckpt_dir)

        self.cnt = latest_step
        
        if hasattr(trainer, 'logger') and trainer.accelerator.is_main_process:
            trainer.logger.info(f'Resumed training from checkpoint: {latest_ckpt_dir}')
    
    
    def on_epoch_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        
        self.cnt += 1

        ckpt_path = self.folder_path / f'ckpt_epoch_{self.cnt}'
        trainer.accelerator.save_state(ckpt_path, safe_serialization=False)
        
        if hasattr(trainer, 'logger') and trainer.accelerator.is_main_process:
            trainer.logger.info(f'Saved checkpoint at: {ckpt_path}')

        
