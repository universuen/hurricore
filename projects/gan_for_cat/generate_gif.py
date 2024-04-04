import _path_setup  # noqa: F401

import torch
import imageio
from tqdm import tqdm

from models import Generator
from configs.for_256px import GeneratorConfig, PathConfig

CKPT_PATH = PathConfig().checkpoints / 'ckpt_step_900'

if __name__ == '__main__':
    # load model
    generator = Generator(**GeneratorConfig()).to('cpu')
    model_path = CKPT_PATH / 'pytorch_model.bin'
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()
    # create interpolation
    sparse_z = torch.randn(10, GeneratorConfig().z_dim)
    sparse_z = torch.cat([sparse_z, sparse_z[0].unsqueeze(0)])
    interploated_z = []
    for i in range(len(sparse_z) - 1):
        for j in range(10):
            interploated_z.append(sparse_z[i] + (sparse_z[i + 1] - sparse_z[i]) * j / 10)
    interploated_z.append(sparse_z[-1])
    # generate images
    all_images = []
    for z in tqdm(interploated_z):
        z = z.to('cpu')
        with torch.no_grad():
            image = generator(z.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image + 1) / 2
        image = (image * 255).astype('uint8')
        all_images.append(image)
    # save gif
    imageio.mimsave(PathConfig().data / 'interploated.gif', all_images, fps=24)
    print(f"Saved gif to {PathConfig().data / 'interploated.gif'}.")
    