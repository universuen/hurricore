import path_setup

import torch
import torchvision
from torch.optim import AdamW
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
from accelerate import Accelerator

from hurricane.logger import Logger
from hurricane.utils import launch

from configs import TrainingConfig, UpdatedPathConfig, LoggerConfig, AcceleratorConfig, CKPTConfig
from resnet_trainer import ResNetTrainer


def main():
    accelerator_config = AcceleratorConfig()
    logger_config = LoggerConfig()
    training_config = TrainingConfig()
    path_config = UpdatedPathConfig()
    ckpt_config = CKPTConfig()
    
    accelerator = Accelerator(**accelerator_config)
    logger = Logger('train_resnet18_on_cifar10', **logger_config)

    if accelerator.is_main_process:
        logger.info(accelerator_config)
        logger.info(logger_config)
        logger.info(training_config)
        logger.info(path_config)
        logger.info(ckpt_config)
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    with accelerator.main_process_first():
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
    
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = AdamW(
        params=model.parameters(), 
        lr=training_config.lr,
    )
    
    trainer = ResNetTrainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        logger=logger,
        ckpt_folder_path=ckpt_config.folder_path,
    )
    trainer.run(epochs=training_config.epochs)


if __name__ == '__main__':
    launch(main, num_processes=1, use_port="8000")
