from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricore.utils import Logger, launch, import_config

from models import Generator, Discriminator
from cat_dataset import CatDataset
from gan_trainer import GANTrainer

# import config from module path
config = import_config('configs.for_256px')

""" Optional:
import config from file path
`config = import_config('examples/gan_for_cat/configs/for_256px.py')`
import config from url
`config = import_config('https://raw.githubusercontent.com/universuen/hurricore/main/examples/gan_for_cat/configs/for_256px.py')`
"""


def main():
    # setup logger and accelerator
    logger = Logger(**config.LoggerConfig())
    accelerator = Accelerator(**config.AcceleratorConfig())
    # setup models
    g_model = Generator(**config.GeneratorConfig())
    d_model = Discriminator(**config.DiscriminatorConfig())
    # setup dataset and dataloader
    with accelerator.main_process_first():
        dataset = CatDataset(**config.DatasetConfig())
    data_loader = DataLoader(dataset, **config.DataLoaderConfig())
    # setup optimizers and lr schedulers
    g_optimizer = AdamW(g_model.parameters(), **config.GeneratorOptimizerConfig()) 
    d_optimizer = AdamW(d_model.parameters(), **config.DiscriminatorOptimizerConfig())
    num_steps_per_epoch = len(data_loader)
    num_epochs = config.TrainerConfig().num_epochs
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    g_scheduler = CosineAnnealingLR(
        optimizer=g_optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    d_scheduler = CosineAnnealingLR(
        optimizer=d_optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    # setup trainer and run
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
        **config.TrainerConfig(),
    )
    trainer.run()

launch(main, **config.LaunchConfig())
