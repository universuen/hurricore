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

    def on_epoch_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=(trainer.ctx.epoch - 1) * len(trainer.data_loader),
        )
        trainer.tb_writer = self.writer
        
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()
        idx = trainer.ctx.batch_idx + 1
        if trainer.accelerator.is_main_process and idx % self.interval == 0:
            num_batches = len(trainer.data_loader)
            step = (trainer.ctx.epoch - 1) * num_batches + idx
            self.writer.add_scalar('Loss/Training', loss, step)
            self.writer.add_scalar('Learning Rate', trainer.optimizer.param_groups[0]['lr'], step)
            self.writer.flush()
    
    def on_epoch_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        if trainer.accelerator.is_main_process:
            idx = trainer.ctx.batch_idx + 1
            num_batches = len(trainer.data_loader)
            step = (trainer.ctx.epoch - 1) * num_batches + idx
            for layer_name, value in trainer.model.named_parameters():
                    if value.grad is not None:
                        self.writer.add_histogram(f"Gradients/{layer_name}", value.grad, step)
            self.writer.flush()
    
    def on_epoch_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        self.writer.close()
