import _path_setup

from torch.utils.data import DataLoader
from torch.optim import RMSprop
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
    g_optimizer = RMSprop(g_model.parameters(), **GeneratorOptimizerConfig()) 
    d_optimizer = RMSprop(d_model.parameters(), **DiscriminatorOptimizerConfig())
    trainer = GANTrainer(
        g_model=g_model,
        d_model=d_model,
        data_loader=data_loader, 
        g_optimizer=g_optimizer, 
        d_optimizer=d_optimizer, 
        accelerator=accelerator,
        logger=logger,
        **TrainerConfig(),
    )
    trainer.run()

launch(main, **LaunchConfig())
