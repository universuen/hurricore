from pathlib import Path

from hurricore.hooks.hurricore_hook import HurricoreHook
from hurricore.utils.huggingface_model_saver import HuggingfaceModelSaver
from hurricore.utils.environment import env


class HuggingfaceModelSaveHook(HurricoreHook):
    def __init__(
        self, 
        trainer, 
        folder_path: Path,
        interval: int = 100,
    ) -> None:
        super().__init__(
            trainer=trainer,
            saving_path=folder_path,
            interval=interval,
        )
        self.folder_path = folder_path
        self.interval = interval
        self.model_saver = HuggingfaceModelSaver(
            trainer=trainer,
            folder_path=folder_path,
        )
        env.huggingface_model_saver = self.model_saver

    def execute_on_step_end(self):
        step = self.trainer.training_context.global_step + 1
        if step % self.interval == 0 or step == self.trainer.training_context.total_steps:
            name = f'ckpt_step_{step}'
            self.model_saver.save(name)
