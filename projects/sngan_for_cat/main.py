import _path_setup

from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

from hurricane.utils import launch, log_all_configs
from hurricane.logger import Logger

from cat_dataset import CatDataset
from sngan import SNGAN
from sngan_trainer import SNGANTrainer
from configs.default import *


def main():
    
    logger = Logger(**LoggerConfig())
    log_all_configs(logger)
    
    accelerator = Accelerator(**AcceleratorConfig())
    
    model = SNGAN(**SNGANConfig())
    
    with accelerator.main_process_first():
        dataset = CatDataset()
    data_loader = DataLoader(dataset, **DataLoaderConfig())
    
    g_optimizer = AdamW(model.generator.parameters())
    d_optimizer = AdamW(model.discriminator.parameters())
    
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
