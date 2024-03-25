from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer
from hurricane.trainers.trainer_base import TrainerBase


class TensorBoardHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        folder_path: Path = None,
        interval: int = 1,
    ) -> None:
        super().__init__(trainer)
        self.is_available = (folder_path is not None and folder_path.is_dir())
        if not self.is_available:
            return
        self.log_dir = folder_path
        self.interval = interval
        self.trainer.tb_log_dir = folder_path
    
    def on_step_end(self) -> None:
        if not self.is_available:
            return
        step = self.trainer.ctx.global_step
        if step % self.interval == 0:
            loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                with SummaryWriter(
                    log_dir=self.log_dir,
                    purge_step=step,
                ) as writer:
                    writer.add_scalar('Loss/Training', loss, step)
                    writer.add_scalar('Learning Rate', self.trainer.optimizer.param_groups[0]['lr'], step)
    
    def on_epoch_end(self) -> None:
        if not self.is_available:
            return
        if self.trainer.accelerator.is_main_process:
            step = self.trainer.ctx.global_step
            with SummaryWriter(
                log_dir=self.log_dir,
                purge_step=step,
            ) as writer:
                for layer_name, value in self.trainer.model.named_parameters():
                        if value.grad is not None:
                            writer.add_histogram(f"Gradients/{layer_name}", value.grad, step)
