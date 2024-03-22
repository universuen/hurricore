from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer


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
        
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()
        step = trainer.ctx.global_step
        if trainer.accelerator.is_main_process and step % self.interval == 0:
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
