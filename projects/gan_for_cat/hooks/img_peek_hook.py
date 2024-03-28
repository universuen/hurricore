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
        assert interval > 0, 'Image peek interval must be greater than 0.'
        assert folder_path is not None and folder_path.is_dir(), 'Invalid image peek folder path.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        self.folder_path = folder_path
        self.peek_interval = interval
        z = torch.randn(4, trainer.originals.model.z_dim, 1, 1)
        trainer.ctx.z = z
        
    def on_step_end(self):
        conditions = (
            self.trainer.accelerator.is_main_process,
            is_deepspeed_zero3(self.trainer.accelerator),
            self.trainer.ctx.global_step % self.peek_interval == 0
        )
        if any(conditions[:2]) and conditions[2]:
            self.trainer.g_model.eval()
            images = self.trainer.g_model(self.trainer.ctx.z.to(self.trainer.accelerator.device))
            image_grid = make_grid(images, nrow=2)
            filename = self.folder_path / f"results_at_step_{self.trainer.ctx.global_step}.png"
            save_image(image_grid, filename)
            if hasattr(self.trainer, 'logger'):
                self.trainer.logger.info(f'Generated images saved at {filename}')
