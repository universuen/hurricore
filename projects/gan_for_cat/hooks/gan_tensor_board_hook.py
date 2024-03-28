from torch.utils.tensorboard import SummaryWriter

from hurricane.hooks.tensor_board_hook import TensorBoardHook


class GANTensorBoardHook(TensorBoardHook):
    def on_step_end(self) -> None:
        step = self.trainer.ctx.global_step
        if step % self.interval == 0:
            g_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.g_step_loss).detach().mean().item()
            d_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.d_step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                writer = self.get_temp_writer()
                writer.add_scalar('Loss/Generator', g_step_loss, step)
                writer.add_scalar('Loss/Discriminator', d_step_loss, step)
                writer.add_scalar('Learning Rate/Generator', self.trainer.g_optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('Learning Rate/Discriminator', self.trainer.d_optimizer.param_groups[0]['lr'], step)
                writer.close()
