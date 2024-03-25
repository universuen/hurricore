from torch.cuda import memory_reserved

from hurricane.hooks.logger_hook import LoggerHook
from hurricane.utils import get_list_mean


class GANLoggerHook(LoggerHook):
    def on_step_end(self) -> None:
        if not self.is_available:
            return
        self.step += 1
        idx = self.trainer.ctx.batch_idx
        num_batches = len(self.trainer.data_loader)
        if self.trainer.ctx.global_step % self.interval == 0 or idx == num_batches:
            g_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.g_step_loss).detach().mean().item()
            d_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.d_step_loss).detach().mean().item()
            if self.trainer.accelerator.is_main_process:
                self.losses_per_batch.append((g_step_loss, d_step_loss))
                idx = self.trainer.ctx.batch_idx
                epoch = self.trainer.ctx.epoch
                num_batches = len(self.trainer.data_loader)
                progress = idx / num_batches
                remaining_time = self._get_remaining_time(num_batches, idx)
                
                self.logger.info(
                    f"Epoch: {epoch}/{self.trainer.epochs} | "
                    f"Step: {idx}/{num_batches} | "
                    f"Generator loss: {g_step_loss:.5f} | "
                    f"Discriminator loss: {d_step_loss:.5f} |"
                    f"Progress: {progress:.2%} | "
                    f"Time left: {remaining_time} | "
                    f"Generator LR: {self.trainer.g_optimizer.param_groups[0]['lr']} | "
                    f"Discriminator LR: {self.trainer.d_optimizer.param_groups[0]['lr']} | "
                    f"Memory used: {memory_reserved() / 1024 ** 3:.2f}GB"
                )

    def on_epoch_end(self) -> None:
        if self.trainer.accelerator.is_main_process:
            g_losses, d_losses = list(zip(*self.losses_per_batch))
            avg_g_loss = get_list_mean(g_losses)
            avg_d_loss = get_list_mean(d_losses)
            self.logger.info(f'Epoch {self.trainer.ctx.epoch} finished with average generator loss: {avg_g_loss} and average discriminator loss: {avg_d_loss}')
