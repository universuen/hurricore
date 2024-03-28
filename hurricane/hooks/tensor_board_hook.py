from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks import HookBase
from hurricane.trainers import TrainerBase


class TensorBoardHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        folder_path: Path = None,
        interval: int = 1,
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert interval > 0, 'TensorBoard interval must be greater than 0.'
        assert folder_path is not None and folder_path.is_dir(), 'Invalid TensorBoard folder path.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        # setup self
        self.interval = interval
        self.folder_path = folder_path
    
    def get_temp_writer(self):
        # use temperary writer to avoid step conflict
        return SummaryWriter(
            log_dir=self.folder_path,
            purge_step=self.trainer.ctx.global_step,
        )      
            
    def on_step_end(self) -> None:
        step = self.trainer.ctx.global_step
        if step % self.interval == 0:
            loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                writer = self.get_temp_writer()
                writer.add_scalar('Loss/Training', loss, step)
                writer.close()
                   
    def on_epoch_end(self) -> None:
        if self.trainer.accelerator.is_main_process:
            writer = self.get_temp_writer()
            step = self.trainer.ctx.global_step
            for layer_name, param in self.trainer.model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Parameters/{layer_name}", param, step)
                    writer.add_histogram(f"Gradients/{layer_name}", param.grad, step)
            writer.close()
