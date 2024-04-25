import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from hurricore.utils import import_config

from unet import UNet
from noise_schedulers import DDPMNoiseScheduler


if __name__ == '__main__':
    torch.manual_seed(0)
    config = import_config('configs.ddpm_128px')
    noise_scheduler = DDPMNoiseScheduler(**config.DDPMNoiseSchedulerConfig()).to('cuda')
    model = UNet(**config.UNetConfig())
    model.load_state_dict(torch.load(config.PathConfig().checkpoints / 'ckpt_step_44500' / 'pytorch_model.bin'))
    model.eval()
    model = model.cuda()
    
    images = torch.randn(9, 3, config.image_size, config.image_size).cuda()
    with torch.no_grad():
        for t in tqdm(list(reversed(range(noise_scheduler.num_steps)))):
            t = torch.full((9, ), t, dtype=torch.long).cuda()
            noise = model.forward(images, t)
            images = noise_scheduler.recover(images, noise, t)
    
    save_image(make_grid(images, nrow=3), config.PathConfig().data / 'sample.png')
    print(f'Saved sample image to {config.PathConfig().data / "sample.png"}')
