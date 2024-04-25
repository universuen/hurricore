from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricore.utils import Logger, launch, import_config


# import config from module path
config = import_config('configs.debug')

""" Optional:
import config from file path
`config = import_config('projects/_project_template/configs/default.py')`
import config from url
`config = import_config('https://raw.githubusercontent.com/universuen/hurricane/main/projects/_project_template/configs/default.py')`
"""


def main():
    # setup logger and accelerator
    logger = Logger(**config.LoggerConfig())
    accelerator = Accelerator(**config.AcceleratorConfig())
    # setup dataset, model and dataloader
    with accelerator.main_process_first():
        dataset = ...
        model = ...
    data_loader = DataLoader(
        dataset=dataset, 
        **config.DataLoaderConfig(),
    )
    # setup optimizer and lr scheduler
    optimizer = AdamW(
        params=model.parameters(), 
        **config.OptimizerConfig(),
    )
    num_steps_per_epoch = len(data_loader)
    num_epochs = config.TrainerConfig().num_epochs
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    # setup trainer and run
    trainer = ...
    trainer.run()

if __name__ == '__main__':
    launch(main, **config.LaunchConfig())
