import _path_setup

from torch.utils.data import DataLoader
from torch.optim import RMSprop
from accelerate import Accelerator

from hurricane.utils import launch, log_all_configs
from hurricane.logger import Logger

from cat_dataset import CatDataset
from projects.gan_for_cat.gan import GAN
from projects.gan_for_cat.gan_trainer import GANTrainer
from configs.default import *


def main():
    logger = Logger(**LoggerConfig())
    accelerator = Accelerator(**AcceleratorConfig())
    if accelerator.is_main_process:
        log_all_configs(logger)
    
    model = GAN(**GANConfig())
    
    with accelerator.main_process_first():
        dataset = CatDataset(**DatasetConfig())
    data_loader = DataLoader(dataset, **DataLoaderConfig())
    
    g_optimizer = RMSprop(model.generator.parameters(), **GeneratorOptimizerConfig()) 
    d_optimizer = RMSprop(model.discriminator.parameters(), **DiscriminatorOptimizerConfig())
    
    trainer = GANTrainer(
        model=model, 
        data_loader=data_loader, 
        g_optimizer=g_optimizer, 
        d_optimizer=d_optimizer, 
        accelerator=accelerator,
        logger=logger,
        **TrainerConfig(),
    )
    trainer.run()

launch(main, num_processes=4, use_port='8000')
