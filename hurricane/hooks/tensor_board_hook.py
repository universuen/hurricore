from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.event_pb2 import Event, SessionLog

from hurricane.hooks import HookBase
from hurricane.trainers import TrainerBase
from hurricane.utils import auto_name


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
        self.writer = SummaryWriter(
            log_dir=self.folder_path,
            purge_step=0,
        )
    
    def on_step_end(self) -> None:
        step = self.trainer.ctx.global_step 
        if (step + 1) % self.interval == 0:
            loss = self.trainer.accelerator.gather(self.trainer.ctx.step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                self.writer.add_scalar('Loss/Training', loss, step)
                self.writer.flush()
                   
    def on_epoch_end(self) -> None:
        if self.trainer.accelerator.is_main_process:
            step = self.trainer.ctx.global_step
            models = self.trainer.originals.models
            for model_name, model in zip(auto_name(models), models):
                for layer_name, param in model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"Parameters/{model_name}-{layer_name}", param, step)
                        self.writer.add_histogram(f"Gradients/{model_name}-{layer_name}", param.grad, step)
            self.writer.flush()

    def on_training_end(self) -> None:
        self.writer.close()
    
    def recover_from_checkpoint(self) -> None:
        self.writer.purge_step = self.trainer.ctx.global_step
        self.writer.close()
        pass