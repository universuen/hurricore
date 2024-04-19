import _path_setup  # noqa: F401

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricane.utils import Logger, launch, import_config

from cat_dog_dataset import CatDogDataset
from noise_cat_dataset import NoiseCatDataset
from flow_trainer import FlowTrainer
from unet import UNet

# import config from module path
config = import_config('configs.cat_generation')

""" Optional:
import config from file path
`config = import_config('projects/_project_template/configs/default.py')`
import config from url
`config = import_config('https://raw.githubusercontent.com/universuen/hurricane/main/projects/_project_template/configs/default.py')`
"""


def main():
    # setup logger and accelerator
    logger = Logger(**config.LoggerConfig())
    accelerator = Accelerator(**config.AcceleratorConfig())
    # setup dataset, model and dataloader
    with accelerator.main_process_first():
        if config.config_name == 'cat_to_dog':
            training_dataset = CatDogDataset(**config.TrainingCatDogDatasetConfig())
            validation_dataset = CatDogDataset(**config.ValidationCatDogDatasetConfig())
        elif config.config_name == 'cat_generation':
            training_dataset = NoiseCatDataset(**config.TrainingNoiseCatDatasetConfig())
            validation_dataset = NoiseCatDataset(**config.ValidationNoiseCatDatasetConfig())
        model = UNet(**config.UNetConfig())
    training_data_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, 
        **config.DataLoaderConfig(),
    )
    # setup optimizer and lr scheduler
    optimizer = AdamW(
        params=model.parameters(), 
        **config.OptimizerConfig(),
    )
    num_steps_per_epoch = len(training_data_loader)
    num_epochs = config.FlowTrainerConfig().num_epochs
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    # setup trainer and run
    trainer = FlowTrainer(
        model=model,
        training_data_loader=training_data_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        lr_scheduler=lr_scheduler,
        logger=logger,
        img_peek_dataset=validation_dataset,
        **config.FlowTrainerConfig(),
    )
    trainer.run()

if __name__ == '__main__':
    launch(main, **config.LaunchConfig())
