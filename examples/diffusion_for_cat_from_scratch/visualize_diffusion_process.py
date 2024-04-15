import _path_setup  # noqa: F401

import torch
from torchvision.utils import make_grid
import imageio
from tqdm import tqdm

from hurricane.utils import import_config
from cat_dataset import CatDataset
from noise_schedulers import DDPMNoiseScheduler


if __name__ == '__main__':
    config = import_config('configs.ddpm_256px')
    ddpm_noise_scheduler = DDPMNoiseScheduler(**config.DDPMNoiseSchedulerConfig())
    dataset = CatDataset(**config.DatasetConfig())
    images = torch.stack([dataset[i] for i in range(9)])
    data_path = config.PathConfig().data

    corrupted_images_sequence = []
    noises_sequence = []
    corrupted_images_sequence_for_gif = []
    for t in tqdm(range(100)):
        t = torch.full((images.shape[0], ), t, dtype=torch.long)
        corrupted_images, noises = ddpm_noise_scheduler.corrupt(images, t)
        corrupted_images_sequence.append(corrupted_images.clone())
        noises_sequence.append(noises.clone())
        if (t[0] + 1) % 10 == 0:
            corrupted_images = (corrupted_images + 1) / 2
            corrupted_images = torch.clamp(corrupted_images, 0, 1)
            corrupted_images = make_grid(corrupted_images.cpu(), nrow=3).permute(1, 2, 0).numpy()
            corrupted_images_sequence_for_gif.append((corrupted_images * 255).astype('uint8'))


    print('Recovering images...')
    recovered_images_sequence_for_gif = []
    corrupted_images = corrupted_images_sequence[-1]
    for t in tqdm(reversed(range(100))):
        noises = noises_sequence[t]
        t = torch.full((images.shape[0], ), t, dtype=torch.long)
        recovered_images = ddpm_noise_scheduler.recover(corrupted_images, noises, t)
        corrupted_images = recovered_images
        if (t[0] + 1) % 10 == 0:
            recovered_images = (recovered_images + 1) / 2
            recovered_images = torch.clamp(recovered_images, 0, 1)
            recovered_images = make_grid(recovered_images.cpu(), nrow=3).permute(1, 2, 0).numpy()
            recovered_images_sequence_for_gif.append((recovered_images * 255).astype('uint8'))
        
    print('Saving GIFs...')
    print(f'Length of corrupted_images_sequence_for_gif: {len(corrupted_images_sequence_for_gif)}')
    print(f'Length of recovered_images_sequence_for_gif: {len(recovered_images_sequence_for_gif)}')
    imageio.mimsave(data_path / 'corrupted_images.gif', corrupted_images_sequence_for_gif, fps=10)
    imageio.mimsave(data_path / 'recovered_images.gif', recovered_images_sequence_for_gif, fps=10)
    print(f'Saved corrupted images to {data_path / "corrupted_images.gif"}')
    print(f'Saved recovered images to {data_path / "recovered_images.gif"}')
