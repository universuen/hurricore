from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer
from hurricane.trainers.trainer_base import TrainerBase


class TensorBoardHook(HookBase):
    def __init__(
        self, 
        folder_path: Path = None,
        interval: int = 1,
    ) -> None:
        super().__init__()
        self.is_available = (folder_path is not None and folder_path.is_dir())
        if not self.is_available:
            return
        self.log_dir = folder_path
        self.interval = interval
    
    def on_training_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        trainer.ctx.tb_log_dir = self.log_dir
    
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        step = trainer.ctx.global_step
        if step % self.interval == 0:
            loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()
            if trainer.accelerator.is_main_process:
                with SummaryWriter(
                    log_dir=self.log_dir,
                    purge_step=step,
                ) as writer:
                    writer.add_scalar('Loss/Training', loss, step)
                    writer.add_scalar('Learning Rate', trainer.optimizer.param_groups[0]['lr'], step)
    
    def on_epoch_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        if trainer.accelerator.is_main_process:
            step = trainer.ctx.global_step
            with SummaryWriter(
                log_dir=self.log_dir,
                purge_step=step,
            ) as writer:
                for layer_name, value in trainer.model.named_parameters():
                        if value.grad is not None:
                            writer.add_histogram(f"Gradients/{layer_name}", value.grad, step)
