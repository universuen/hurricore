from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.tensor_board_hook import TensorBoardHook
from hurricane.trainers.trainer import Trainer


class SNGANTensorBoardHook(TensorBoardHook):
    def on_step_end(self, trainer: Trainer) -> None:
        if not self.is_available:
            return
        step = trainer.ctx.global_step
        if step % self.interval == 0:
            g_step_loss = trainer.accelerator.gather(trainer.ctx.g_step_loss).detach().mean().item()
            d_step_loss = trainer.accelerator.gather(trainer.ctx.d_step_loss).detach().mean().item()
            if trainer.accelerator.is_main_process:
                with SummaryWriter(
                    log_dir=self.log_dir,
                    purge_step=step,
                ) as writer:
                    writer.add_scalar('Loss/Training/Generator', g_step_loss, step)
                    writer.add_scalar('Loss/Training/Discriminator', d_step_loss, step)
                    writer.add_scalar('Generator LR', trainer.g_optimizer.param_groups[0]['lr'], step)
                    writer.add_scalar('Discriminator LR', trainer.d_optimizer.param_groups[0]['lr'], step)
