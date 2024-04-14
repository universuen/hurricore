import _path_setup  # noqa: F401

import torch
from torchvision.utils import make_grid
import imageio
from tqdm import tqdm

from hurricane.utils import import_config

from noise_schedulers import DDPMNoiseScheduler
from cat_dataset import CatDataset

if __name__ == '__main__':
    config = import_config('configs.ddpm_256px')
    noise_scheduler = DDPMNoiseScheduler(**config.DDPMNoiseSchedulerConfig())
    dataset = CatDataset(**config.DatasetConfig())
    images = torch.stack([dataset[i] for i in range(4)]).cuda()
    data_path = config.PathConfig().data

    corrupted_images_sequence = []
    noises_sequence = []
    for t in tqdm(range(1000)):
        t = torch.ones(images.shape[0], dtype=torch.int) * t
        corrupted_images, noises = noise_scheduler.corrupt(images, t)
        noises_sequence.append(noises)
        corrupted_images = (corrupted_images + 1) / 2
        corrupted_images = make_grid(corrupted_images.cpu(), nrow=2).permute(1, 2, 0).numpy()
        corrupted_images_sequence.append((corrupted_images * 255).astype('uint8'))

    recovered_images_sequence = []
    corrupted_images = corrupted_images_sequence[-1]
    noises_sequence.reverse()
    for t, noises in tqdm(enumerate(noises_sequence)):
        t = 1000 - t - 1
        t = torch.ones(images.shape[0], dtype=torch.int) * t
        recovered_images = noise_scheduler.recover(corrupted_images, noises, t)
        recovered_images = (recovered_images + 1) / 2
        recovered_images = make_grid(recovered_images.cpu(), nrow=2).permute(1, 2, 0).numpy()
        recovered_images_sequence.append((recovered_images * 255).astype('uint8'))
    print('Saving GIFs...')
    imageio.mimsave(data_path / 'corrupted_images.gif', corrupted_images_sequence, fps=30)
    imageio.mimsave(data_path / 'recovered_images.gif', recovered_images_sequence, fps=30)
    print('GIFs saved successfully.')
