from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.hook_base import HookBase
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.trainers.trainer import Trainer


class TensorBoardHook(HookBase):
    def __init__(self, folder_path: Path = None) -> None:
        super().__init__()
        self.is_available = (folder_path is not None and folder_path.is_dir())
        if not self.is_available:
            return
        self.writer = SummaryWriter(log_dir=folder_path)
                
    
    def on_training_start(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        trainer.tb_writer = self.writer
        if trainer.accelerator.is_main_process:
            with torch.no_grad():
                self.writer.add_graph(
                    model=trainer.accelerator.unwrap_model(trainer.model), 
                    input_to_model=next(iter(trainer.data_loader))[0],
                )
            self.writer.flush()
    
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        loss = trainer.accelerator.gather(trainer.ctx.step_loss).detach().mean().item()
        if trainer.accelerator.is_main_process:
            num_batches = len(trainer.data_loader)
            step = (trainer.ctx.epoch - 1) * num_batches + trainer.ctx.batch_idx + 1
            self.writer.add_scalar('Loss/Training', loss, step)
            self.writer.add_scalar('Learning Rate', trainer.optimizer.param_groups[0]['lr'], step)
            for layer_name, value in trainer.model.named_parameters():
                if value.grad is not None:
                    self.writer.add_histogram(f"Gradients/{layer_name}", value.grad.cpu(), step)
            self.writer.flush()
    
    def on_training_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        self.writer.close()
