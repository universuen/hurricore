from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.tensor_board_hook import TensorBoardHook


class GANTensorBoardHook(TensorBoardHook):
    def on_step_end(self) -> None:
        if not self.is_available:
            return
        step = self.trainer.ctx.global_step
        if step % self.interval == 0:
            g_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.g_step_loss).detach().mean().item()
            d_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.d_step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                with SummaryWriter(
                    log_dir=self.log_dir,
                    purge_step=step,
                ) as writer:
                    writer.add_scalar('Loss/Training/Generator', g_step_loss, step)
                    writer.add_scalar('Loss/Training/Discriminator', d_step_loss, step)
                    writer.add_scalar('Generator LR', self.trainer.g_optimizer.param_groups[0]['lr'], step)
                    writer.add_scalar('Discriminator LR', self.trainer.d_optimizer.param_groups[0]['lr'], step)
