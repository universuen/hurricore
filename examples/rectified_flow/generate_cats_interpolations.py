import _path_setup  # noqa: F401

import torch
from torchvision.utils import make_grid
import numpy as np
import imageio
from rich.progress import track

from hurricane.utils import import_config, find_latest_checkpoint

from unet import UNet
from navigator import Navigator


CONFIG = "configs.cat_generation"
NUM_4_GRID_CATS = 5
STEP = 30


if __name__ == "__main__":
    config = import_config(CONFIG)
    model = UNet(**config.UNetConfig()).cuda()
    latest_ckpt_path = find_latest_checkpoint(config.PathConfig().checkpoints)
    print(f'Loading checkpoint from {latest_ckpt_path}')
    state_dict = torch.load(latest_ckpt_path / "pytorch_model.bin")
    model.load_state_dict(state_dict)
    model.eval()
    navigator = Navigator(model, num_steps=100)
    # create interpolations
    image_size = config.image_size
    noises = torch.randn(NUM_4_GRID_CATS, 4, 3, image_size, image_size).cuda()
    noises = torch.cat([noises, noises[0].unsqueeze(0)], dim=0)
    images = []
    for i in range(len(noises) - 1):
        noise_a = noises[i]
        noise_b = noises[i + 1]
        for alpha in track(torch.linspace(0, 1, STEP), f"Interpolation {i} -> {i + 1}"):
            noise_interpolation = (1 - alpha) * noise_a + alpha * noise_b
            image = navigator.navigate(noise_interpolation)
            grid_image = make_grid(image, nrow=2)
            np_image = grid_image.permute(1, 2, 0).clamp(-1, 1).squeeze(0).detach().cpu().numpy()
            np_image = (np_image + 1) / 2 * 255 
            np_image = np_image.astype(np.uint8)
            images.append(np_image)
    # save images
    output_path = config.PathConfig().data / "cats_interpolations.gif"
    imageio.mimsave(output_path, images, fps=10)
    print(f"Interpolations saved to {output_path}")
