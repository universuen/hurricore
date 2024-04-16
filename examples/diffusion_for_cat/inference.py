import _path_setup  # noqa: F401

import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from tqdm import tqdm
from PIL import Image

from hurricane.utils import get_total_parameters

from noise_schedulers import DDPMNoiseScheduler

# ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
# image = ddpm(num_inference_steps=1000).images[0]
# image.save("/storage_fast/ysun/hurricane/examples/diffusion_for_cat_from_scratch/cat.png")

# scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
# model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
# scheduler.set_timesteps(1000)
# print(get_total_parameters(model))

# sample_size = model.config.sample_size
# noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")

# for t in tqdm(scheduler.timesteps):
#     with torch.no_grad():
#         noisy_residual = model(noise, t).sample
#     previous_noisy_sample = scheduler.step(noisy_residual, t, noise).prev_sample
#     noise = previous_noisy_sample


# image = (noise / 2 + 0.5).clamp(0, 1).squeeze()
# image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
# image = Image.fromarray(image)
# image.save("/storage_fast/ysun/hurricane/examples/diffusion_for_cat_from_scratch/cat.png")


scheduler = DDPMNoiseScheduler()
# scheduler.to("cuda")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True)# .to("cuda")
print(model)
print(get_total_parameters(model))
exit()

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")

for t in tqdm(reversed(range(scheduler.num_steps))):
    if t == 0:
        pass
    t = torch.tensor(t, device="cuda")
    with torch.no_grad():
        noisy_residual = model(noise, t).sample
    previous_noisy_sample = scheduler.recover(noise, noisy_residual, t)
    noise = previous_noisy_sample


image = (noise / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image.save("/storage_fast/ysun/hurricane/examples/diffusion_for_cat_from_scratch/data/cat.png")

