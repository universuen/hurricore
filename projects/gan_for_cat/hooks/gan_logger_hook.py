from torch.cuda import memory_reserved

from hurricane.hooks.logger_hook import LoggerHook
from hurricane.utils import get_list_mean


class GANLoggerHook(LoggerHook):
    def on_step_end(self) -> None:
        self.step += 1
        idx = self.trainer.ctx.batch_idx
        iterator_length = len(self.trainer.data_loader)
        if self.trainer.ctx.global_step % self.interval == 0 or idx == iterator_length:
            g_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.g_step_loss).detach().mean().item()
            d_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.d_step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                self.losses_per_batch.append((g_step_loss, d_step_loss))
                epoch = self.trainer.ctx.epoch
                progress = idx / iterator_length
                remaining_time = self._get_remaining_time(iterator_length, idx)
                
                self.logger.info(
                    f"Epoch: {epoch}/{self.trainer.epochs} | "
                    f"Step: {idx}/{iterator_length} | "
                    f"Generator loss: {g_step_loss:.5f} | "
                    f"Discriminator loss: {d_step_loss:.5f} |"
                    f"Progress: {progress:.2%} | "
                    f"Time left: {remaining_time} | "
                    f"Memory used: {memory_reserved() / 1024 ** 3:.2f}GB"
                )

    def on_epoch_end(self) -> None:
        if self.trainer.accelerator.is_main_process:
            try:
                g_losses, d_losses = list(zip(*self.losses_per_batch))
                avg_g_loss = get_list_mean(g_losses)
                avg_d_loss = get_list_mean(d_losses)
                self.logger.info(f'Epoch {self.trainer.ctx.epoch} finished with average generator loss: {avg_g_loss} and average discriminator loss: {avg_d_loss}')
            except ValueError as e:
                self.logger.error(e)