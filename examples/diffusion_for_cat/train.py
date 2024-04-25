import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricore.utils import Logger, launch, import_config

from cat_dataset import CatDataset
from unet import UNet
from noise_schedulers import DDPMNoiseScheduler
from diffusion_trainer import DiffusionTrainer

# import config from module path
config = import_config('configs.ddpm_128px')


def main():
    # setup logger and accelerator
    logger = Logger(**config.LoggerConfig())
    accelerator = Accelerator(**config.AcceleratorConfig())
    # setup dataset, model and dataloader
    with accelerator.main_process_first():
        dataset = CatDataset(**config.DatasetConfig())
        model = UNet(**config.UNetConfig())
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        **config.DataLoaderConfig(),
    )
    # setup optimizer and lr scheduler
    optimizer = AdamW(
        params=model.parameters(), 
        **config.OptimizerConfig(),
    )
    num_steps_per_epoch = len(data_loader)
    num_epochs = config.TrainerConfig().num_epochs
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    # setup trainer and run
    trainer = DiffusionTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        accelerator=accelerator,
        noise_scheduler=DDPMNoiseScheduler(**config.DDPMNoiseSchedulerConfig()),
        lr_scheduler=lr_scheduler,
        logger=logger,
        **config.TrainerConfig(),
    )
    trainer.run()

if __name__ == '__main__':
    launch(main, **config.LaunchConfig())
