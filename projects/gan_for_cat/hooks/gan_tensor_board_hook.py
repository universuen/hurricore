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
                writer.add_scalars(
                    main_tag='Loss/Training', 
                    tag_scalar_dict={'Generator': g_step_loss, 'Discriminator': d_step_loss}, 
                    global_step=step
                )
                writer.add_scalars(
                    main_tag='Learning Rate', 
                    tag_scalar_dict={
                        'Generator': self.trainer.g_optimizer.param_groups[0]['lr'], 
                        'Discriminator': self.trainer.d_optimizer.param_groups[0]['lr']
                    }, 
                    global_step=step
                )
