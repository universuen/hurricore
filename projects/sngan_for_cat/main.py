import _path_setup

from torch.utils.data import DataLoader
from torch.optim import RMSprop
from accelerate import Accelerator

from hurricane.utils import launch, log_all_configs
from hurricane.logger import Logger

from cat_dataset import CatDataset
from sngan import SNGAN
from sngan_trainer import SNGANTrainer
from configs.default import *


def main():
    logger = Logger(**LoggerConfig())
    accelerator = Accelerator(**AcceleratorConfig())
    if accelerator.is_main_process:
        log_all_configs(logger)
    
    model = SNGAN(**SNGANConfig())
    
    with accelerator.main_process_first():
        dataset = CatDataset(**DatasetConfig())
    data_loader = DataLoader(dataset, **DataLoaderConfig())
    
    g_optimizer = RMSprop(model.generator.parameters(), **GeneratorOptimizerConfig()) 
    d_optimizer = RMSprop(model.discriminator.parameters(), **DiscriminatorOptimizerConfig())
    
    trainer = SNGANTrainer(
        model=model, 
        data_loader=data_loader, 
        g_optimizer=g_optimizer, 
        d_optimizer=d_optimizer, 
        accelerator=accelerator,
        logger=logger,
        **TrainerConfig(),
    )
    trainer.run()

launch(main, num_processes=1)
