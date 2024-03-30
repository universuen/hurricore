from hurricane.hooks import TensorBoardHook


class GANTensorBoardHook(TensorBoardHook):
    def on_step_end(self) -> None:
        conditions = (
            self.trainer.accelerator.is_main_process,
            (self.trainer.ctx.global_step + 1) % self.interval == 0,
        )
        if all(conditions):
            self.writer.add_scalar('Loss/Generator', self.trainer.ctx.g_step_loss, self.trainer.ctx.global_step)
            self.writer.add_scalar('Loss/Discriminator', self.trainer.ctx.d_step_loss, self.trainer.ctx.global_step)
            self.writer.flush()
