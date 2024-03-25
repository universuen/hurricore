from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from hurricane.hooks.hook_base import HookBase
from hurricane.trainers.trainer_base import TrainerBase
from hurricane.utils import is_deepspeed_zero3


class ImgPeekHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        folder_path: Path,
        interval: int,
    ):
        super().__init__(trainer)
        self.is_available = (folder_path is not None)
        if not self.is_available:
            return
        self.folder_path = folder_path
        self.peek_interval = interval
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        self.z = torch.randn(4, model.z_dim, 1, 1, device=self.trainer.accelerator.device)

    def on_training_start(self) -> None:
        if not self.is_available:
            return
        if self.trainer.accelerator.is_main_process:
            self.folder_path.mkdir(parents=True, exist_ok=True)
            for f in self.folder_path.iterdir():
                f.unlink()
        
    def on_step_end(self):
        conditions = (
            self.trainer.accelerator.is_main_process,
            is_deepspeed_zero3(self.trainer.accelerator),
            self.trainer.ctx.global_step % self.peek_interval == 0
        )
        if any(conditions[:2]) and conditions[2]:
            model = self.trainer.accelerator.unwrap_model(self.trainer.model)
            images = model.generator(self.z)
            image_grid = make_grid(images, nrow=2)
            filename = self.folder_path / f"results_at_step_{self.trainer.ctx.global_step}.png"
            save_image(image_grid, filename)
            if hasattr(self.trainer, 'logger'):
                self.trainer.logger.info(f'Generated images saved at {filename}')
