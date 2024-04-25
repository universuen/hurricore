import _path_setup  # noqa: F401

import torch
import numpy as np
import imageio
from rich.progress import track

from hurricore.utils import import_config, find_latest_checkpoint

from unet import UNet
from navigator import Navigator
from noise_cat_dataset import NoiseCatDataset
from cat_dog_dataset import CatDogDataset


CONFIG = "configs.cat_to_dog"


if __name__ == "__main__":
    config = import_config(CONFIG)
    model = UNet(**config.UNetConfig()).cuda()
    latest_ckpt_path = find_latest_checkpoint(config.PathConfig().checkpoints)
    print(f'Loading checkpoint from {latest_ckpt_path}')
    state_dict = torch.load(latest_ckpt_path / "pytorch_model.bin")
    model.load_state_dict(state_dict)
    model.eval()
    navigator = Navigator(model, num_steps=100)
    
    if "cat_generation" in config.config_name:
        dataset = NoiseCatDataset(**config.ValidationNoiseCatDatasetConfig())
    elif config.config_name == "cat_to_dog":
        dataset = CatDogDataset(**config.ValidationCatDogDatasetConfig())
    else:
        raise ValueError(f"Unknown config_name: {config.config_name}")
    
    image = dataset[0][0].unsqueeze(0).cuda()
    images = []
    for step in track(range(navigator.num_steps), 'forward pass'):
        image = navigator.step(image, step)
        np_image = image.permute(0, 2, 3, 1).clamp(-1, 1).squeeze(0).detach().cpu().numpy()
        np_image = (np_image + 1) / 2 * 255 
        np_image = np_image.astype(np.uint8)
        images.append(np_image)
    imageio.mimsave(config.PathConfig().data / "forward.gif", images, fps=30)
    print(f'Forward image saved to {config.PathConfig().data / "forward.gif"}')
    
    image = dataset[0][1].unsqueeze(0).cuda()
    images = []
    for step in track(range(navigator.num_steps), 'backward pass'):
        image = navigator.step(image, step, reversed=True)
        np_image = image.permute(0, 2, 3, 1).clamp(-1, 1).squeeze(0).detach().cpu().numpy()
        np_image = (np_image + 1) / 2 * 255 
        np_image = np_image.astype(np.uint8)
        images.append(np_image)
    imageio.mimsave(config.PathConfig().data / "backward.gif", images, fps=30)
    print(f'Backward image saved to {config.PathConfig().data / "backward.gif"}')
