from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid

from hurricane.hooks import HookBase, LoggerHook, TensorBoardHook
from hurricane.trainers import TrainerBase

from navigator import Navigator


class ImgPeekHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        dataset: Dataset,
        folder_path: Path,
        interval: int,
    ):
        super().__init__(trainer)
        assert interval > 0, 'Image peek interval must be greater than 0.'
        assert folder_path is not None and folder_path.is_dir(), 'Invalid image peek folder path.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        self.folder_path = folder_path
        self.peek_interval = interval
        src_images = torch.stack([dataset[i][0] for i in range(9)]) 
        trainer.ctx.src_images = src_images
    
    def on_step_end(self):
        if (self.trainer.ctx.global_step + 1) % self.peek_interval == 0:
            model = self.trainer.models[0]
            model.eval()
            with torch.no_grad():
                src_images = self.trainer.ctx.src_images.to(self.trainer.accelerator.device)
                navigator = Navigator(model, num_steps=100)
                tgt_images = navigator.navigate(src_images)
            tgt_images = ((tgt_images + 1) / 2).clamp(0, 1)
            image_grid = make_grid(tgt_images, nrow=3)
            filename = self.folder_path / f"results_at_step_{self.trainer.ctx.global_step + 1}.png"
            if self.trainer.accelerator.is_main_process:
                save_image(image_grid, filename)
            TensorBoardHook.msg_queue.append(
                (
                    'add_image',
                    {
                        'tag': 'Generated Images',
                        'img_tensor': image_grid,
                        'global_step': self.trainer.ctx.global_step + 1,
                    }
                )
            )
            LoggerHook.msg_queue.append(('info', f'Generated images saved at {filename}'))

