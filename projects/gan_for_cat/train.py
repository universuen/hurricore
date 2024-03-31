import _path_setup

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricane.utils import launch, log_all_configs
from hurricane.logger import Logger

from models import Generator, Discriminator
from cat_dataset import CatDataset
from gan_trainer import GANTrainer
from configs.default import *


def main():
    logger = Logger(**LoggerConfig())
    accelerator = Accelerator(**AcceleratorConfig())
    if accelerator.is_main_process:
        log_all_configs(logger)
    g_model = Generator(**GeneratorConfig())
    d_model = Discriminator(**DiscriminatorConfig())
    with accelerator.main_process_first():
        dataset = CatDataset(**DatasetConfig())
    data_loader = DataLoader(dataset, **DataLoaderConfig())
    g_optimizer = AdamW(g_model.parameters(), **GeneratorOptimizerConfig()) 
    d_optimizer = AdamW(d_model.parameters(), **DiscriminatorOptimizerConfig())
    g_scheduler = CosineAnnealingLR(
        optimizer=g_optimizer,
        T_max=(len(data_loader) // AcceleratorConfig().gradient_accumulation_steps) * TrainerConfig().epochs,
    )
    d_scheduler = CosineAnnealingLR(
        optimizer=d_optimizer,
        T_max=(len(data_loader) // AcceleratorConfig().gradient_accumulation_steps) * TrainerConfig().epochs,
    )
    trainer = GANTrainer(
        g_model=g_model,
        d_model=d_model,
        data_loader=data_loader, 
        g_optimizer=g_optimizer, 
        d_optimizer=d_optimizer,
        g_lr_scheduler=g_scheduler,
        d_lr_scheduler=d_scheduler,
        accelerator=accelerator,
        logger=logger,
        **TrainerConfig(),
    )
    trainer.run()

launch(main, **LaunchConfig())
