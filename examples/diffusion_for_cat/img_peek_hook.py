from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from hurricane.hooks import HookBase, LoggerHook, TensorBoardHook
from hurricane.trainers import TrainerBase


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
        image_size = trainer.originals.models[0].image_size
        self.num_steps = trainer.noise_scheduler.num_steps
        z = torch.randn(9, 3, image_size, image_size, device=trainer.accelerator.device)
        trainer.ctx.z = z

    def recover_from_checkpoint(self):
        self.trainer.ctx.z = self.trainer.ctx.z.to(self.trainer.accelerator.device)
    
    def on_step_end(self):
        if (self.trainer.ctx.global_step + 1) % self.peek_interval == 0:
            model = self.trainer.models[0]
            model.eval()
            with torch.no_grad():
                images = self.trainer.ctx.z.to(self.trainer.accelerator.device)
                for t in reversed(range(self.num_steps)):
                    t = torch.full((images.size(0), ), t, device=images.device, dtype=torch.long)
                    predicted_noise = model(images, t)
                    images = self.trainer.noise_scheduler.recover(images, predicted_noise, t)
            images = ((images + 1) / 2).clamp(0, 1)
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

