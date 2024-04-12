from torch.cuda import memory_reserved

from hurricane.hooks.logger_hook import LoggerHook
from hurricane.utils import get_list_mean


class GANLoggerHook(LoggerHook):
    
    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.g_step_losses = []
        self.d_step_losses = []
    
    
    def _collect_step_loss(self):
        g_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.g_step_loss).detach().mean().item()
        d_step_loss = self.trainer.accelerator.gather(self.trainer.ctx.d_step_loss).detach().mean().item()
        self.g_step_losses.append(g_step_loss)
        self.d_step_losses.append(d_step_loss)
    
    
    def _log_states(self):
        idx = self.trainer.ctx.batches_idx + 1
        iterator_length = self.trainer.ctx.iterator_length
        epoch = self.trainer.ctx.epoch + 1
        progress = idx / iterator_length
        remaining_time = self._get_remaining_time()
        
        self.logger.info(
            f"Epoch: {epoch}/{self.trainer.num_epochs} | "
            f"Step: {idx}/{iterator_length} | "
            f"G loss: {self.trainer.ctx.g_step_loss:.5f} | "
            f"D loss: {self.trainer.ctx.d_step_loss:.5f} |"
            f"Progress: {progress:.2%} | "
            f"Time left: {remaining_time} | "
            f"Memory used: {memory_reserved() / 1024 ** 3:.2f}GB"
        )


    def on_epoch_end(self) -> None:
        avg_g_loss = get_list_mean(self.g_step_losses)
        avg_d_loss = get_list_mean(self.d_step_losses)
        self.logger.info(f'Epoch {self.trainer.ctx.epoch + 1} finished')
        self.logger.info(f'Average G loss: {avg_g_loss:.5f}')
        self.logger.info(f'Average D loss: {avg_d_loss:.5f}')

    