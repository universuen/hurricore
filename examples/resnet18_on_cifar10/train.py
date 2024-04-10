import _path_setup  # noqa: F401

import torch
import torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
from accelerate import Accelerator

from hurricane.utils import Logger, launch, log_all_configs, import_config
from resnet_trainer import ResNetTrainer


# import config from module path
config = import_config('configs.default')

""" Optional:
import config from file path
`config = import_config('examples/resnet18_on_cifar10/configs/default.py')`
import config from url
`config = import_config('https://raw.githubusercontent.com/universuen/hurricane/main/examples/resnet18_on_cifar10/configs/default.py')`
"""


def main():
    # setup logger and accelerator
    logger = Logger(**config.LoggerConfig())
    accelerator = Accelerator(**config.AcceleratorConfig())
    if accelerator.is_main_process:
        log_all_configs(logger)
    # setup dataset, model and dataloader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    with accelerator.main_process_first():
        dataset = torchvision.datasets.CIFAR10(
            transform=transform,
            **config.DatasetConfig(),
        )
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
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
    num_epochs = config.TrainerConfig().epochs
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    # setup trainer and run
    trainer = ResNetTrainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        logger=logger,
        lr_scheduler=scheduler,
        lr_scheduler_mode='per_step',
        **config.TrainerConfig(),
    )
    trainer.run()


if __name__ == '__main__':
    launch(main, **config.LaunchConfig())
