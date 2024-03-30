from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from hurricane.hooks import HookBase, LoggerHook, TensorBoardHook
from hurricane.trainers import TrainerBase
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
        z = torch.randn(4, trainer.originals.models[0].z_dim, 1, 1)
        trainer.ctx.z = z
    
    def on_training_start(self) -> None:
        # collect logger
        logger_hook = self.trainer.get_hook(LoggerHook)
        if logger_hook is not None:
            self.logger = logger_hook.logger
        # collect tensor board writer
        tb_hook = self.trainer.get_hook(TensorBoardHook)
        if tb_hook is not None:
            self.writer = tb_hook.writer
    
    def on_step_end(self):
        conditions = (
            self.trainer.accelerator.is_main_process,
            is_deepspeed_zero3(self.trainer.accelerator),
            (self.trainer.ctx.global_step + 1) % self.peek_interval == 0
        )
        if any(conditions[:2]) and conditions[2]:
            g_model = self.trainer.models[0]
            images = g_model(self.trainer.ctx.z.to(self.trainer.accelerator.device))
            image_grid = make_grid(images, nrow=2)
            filename = self.folder_path / f"results_at_step_{self.trainer.ctx.global_step}.png"
            save_image(image_grid, filename)
            if hasattr(self, 'writer'):
                self.writer.add_image('Generated Images', image_grid, self.trainer.ctx.global_step)
            if hasattr(self, 'logger'):
                self.logger.info(f'Generated images saved at {filename}')
