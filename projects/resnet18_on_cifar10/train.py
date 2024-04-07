import _path_setup  # noqa: F401

import torch
import torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
from accelerate import Accelerator

from hurricane.utils import Logger, launch, log_all_configs

from configs.default import (
    LoggerConfig,
    AcceleratorConfig,
    DataLoaderConfig,
    OptimizerConfig,
    TrainerConfig,
    LaunchConfig,
    DatasetConfig,
)
from resnet_trainer import ResNetTrainer


def main():
    logger = Logger(**LoggerConfig())
    accelerator = Accelerator(**AcceleratorConfig())
    if accelerator.is_main_process:
        log_all_configs(logger)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    with accelerator.main_process_first():
        dataset = torchvision.datasets.CIFAR10(
            transform=transform,
            **DatasetConfig(),
        )
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        **DataLoaderConfig(),
    )
    optimizer = AdamW(
        params=model.parameters(), 
        **OptimizerConfig(),
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=(len(data_loader) // AcceleratorConfig().gradient_accumulation_steps) * TrainerConfig().epochs,
    )
    trainer = ResNetTrainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        logger=logger,
        lr_scheduler=scheduler,
        lr_scheduler_mode='per_step',
        **TrainerConfig(),
    )
    trainer.run()

if __name__ == '__main__':
    launch(main, **LaunchConfig())
