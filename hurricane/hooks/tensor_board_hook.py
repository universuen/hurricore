import torch
from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.hook_base import HookBase
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.trainers.trainer import Trainer


class TensorBoardHook(HookBase):
    def __init__(self) -> None:
        super().__init__()
        self.is_available = False
        self.writer: SummaryWriter = None
    
    def _check_availability(self, trainer: Trainer) -> None:
        for hook in trainer.hooks:
            if isinstance(hook, LoggerHook):
                if hasattr(hook.logger, 'logs_dir') and hook.logger.logs_dir is not None:
                    self.is_available = True
                    data_path = hook.logger.logs_dir / 'tensorboard' / hook.logger.name
                    data_path.mkdir(parents=True, exist_ok=True)
                    self.writer = SummaryWriter(data_path)
                    break
                
    
    def on_training_start(self, trainer: Trainer) -> None:
        self._check_availability(trainer)
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
