from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer import Trainer


class ImgPeekHook(HookBase):
    def __init__(self, peek_interval: int):
        super().__init__()
        self.peek_interval = peek_interval

    def on_step_end(self, epoch: int):
        if epoch % self.peek_interval == 0:
            self.trainer.peek(epoch)