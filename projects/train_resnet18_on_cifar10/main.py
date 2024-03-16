import path_setup

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from accelerate import Accelerator

from hurricane.logger import Logger
from hurricane.utils import launch_for_parallel_training
from configs import LoggerConfig

from resnet_trainer import ResNetTrainer

def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, 
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128,
        shuffle=True
    )

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    accelerator = Accelerator()
    logger_config = LoggerConfig()
    logger = Logger('train_restnet18_on_cifar10', **logger_config.to_dict())

    trainer = ResNetTrainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        logger=logger,
    )

    trainer.run(100)

launch_for_parallel_training(main, num_processes=1, use_port="8000")
