from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from hurricane.hooks import Hook, LoggerHook, TensorBoardHook
from hurricane.trainers import Trainer


class ImgPeekHook(Hook):
    def __init__(
        self, 
        trainer: Trainer,
        folder_path: Path,
        interval: int,
    ):
        super().__init__(trainer)
        assert interval > 0, 'Image peek interval must be greater than 0.'
        assert folder_path is not None and folder_path.is_dir(), 'Invalid image peek folder path.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        self.folder_path = folder_path
        self.peek_interval = interval
        z = torch.randn(9, trainer.originals.models[0].z_dim)
        trainer.ctx.z = z

    
    def on_step_end(self):
        if (self.trainer.ctx.global_step + 1) % self.peek_interval == 0:
            g_model = self.trainer.models[0]
            g_model.eval()
            with torch.no_grad():
                images = g_model(self.trainer.ctx.z.to(self.trainer.accelerator.device)).detach()
            images = (images + 1) / 2
            image_grid = make_grid(images, nrow=3)
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

