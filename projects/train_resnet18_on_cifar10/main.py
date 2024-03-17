import path_setup

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
from accelerate import Accelerator

from hurricane.logger import Logger
from hurricane.utils import launch

from configs import TrainingConfig, UpdatedPathConfig, LoggerConfig
from resnet_trainer import ResNetTrainer


def main():
    logger_config = LoggerConfig()
    logger = Logger('train_restnet18_on_cifar10', **logger_config)
    logger.info(logger_config)
    
    path_config = UpdatedPathConfig()
    logger.info(path_config)
    
    training_config = TrainingConfig()
    logger.info(training_config)
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=path_config.cifar10_dataset, 
        train=True,
        download=True, 
        transform=transform,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=training_config.batch_size,
        shuffle=True
    )
    
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = training_config.optimizer_type(
        params=model.parameters(), 
        lr=training_config.lr,
    )
    
    accelerator = Accelerator()
    
    trainer = ResNetTrainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        logger=logger,
    )
    trainer.run(epochs=training_config.epochs)


if __name__ == '__main__':
    launch(main, num_processes=1, use_port="8000")
