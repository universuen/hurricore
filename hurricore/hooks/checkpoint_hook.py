from pathlib import Path

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import env, find_latest_checkpoint_folder, Checkpointer


class CheckpointHook(HurricoreHook):
    def __init__(
        self, 
        trainer: HurricoreTrainer,
        folder_path: Path = None,
        interval: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(
            trainer=trainer,
            folder_path=folder_path,
            interval=interval,
            seed=seed,
        )
        # Setup self attributes
        self.interval = interval
        self.folder_path = Path(folder_path)
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.seed = seed
    

    def execute_on_trainer_init_end(self) -> None:
        # Setup checkpointer
        self.checkpointer = Checkpointer(
            accelerator=self.trainer.accelerator,
            folder_path=self.folder_path,
        )
        # Re-prepare dataloader with seedable sampler if necessary
        self.trainer.data_loaders = [
            self.checkpointer.reprepare_data_loader(dl, self.seed)
            for dl in self.trainer.originals.data_loaders
        ]
        # Register trainer context for checkpointing
        self.checkpointer.register_state(self.trainer.training_context)
        env.checkpointer = self.checkpointer
        
    
    def execute_on_training_start(self) -> None:
        # Check available checkpoint
        ckpt_paths = [
            d for d in self.folder_path.iterdir() 
            if d.is_dir() and d.name.startswith('ckpt_step_')
        ]
        if len(ckpt_paths) == 0:
            return 
        # Load latest checkpoint
        latest_ckpt_path = find_latest_checkpoint_folder(self.folder_path)
        env.logger.info(f'Loading checkpoint: {latest_ckpt_path}')
        self.checkpointer.load_state(latest_ckpt_path)
        # Step into the next batch
        self.trainer.training_context.batch_idx += 1
        # Recover dataloaders states
        for dl in self.trainer.data_loaders:
            dl.set_epoch(self.trainer.training_context.epoch)
            dl.skip_batches = self.trainer.training_context.batch_idx
        # Recover hooks
        self.checkpointer.excute_actions()
        self.trainer.accelerator.wait_for_everyone()
        # Log
        env.logger.info(f'Resumed training from checkpoint: {latest_ckpt_path}')
    
    
    def execute_on_step_end(self) -> None:
        step = self.trainer.training_context.global_step + 1
        if step % self.interval == 0 or step == self.trainer.training_context.total_steps:
            name = f'ckpt_step_{step}'
            self.save_checkpoint(name)


    def save_checkpoint(self, name: str) -> None:
        ckpt_path = self.checkpointer.save_state(name)
        env.logger.info(f'Checkpoint saved at: {ckpt_path}')


    def execute_on_epoch_end(self) -> None:
        for dl in self.trainer.data_loaders:
            dl.skip_batches = 0
