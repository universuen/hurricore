import _path_setup  # noqa: F401

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricane.utils import Logger, launch, log_all_configs, import_config

# import config from module path
config = import_config('configs.default')

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
    if accelerator.is_main_process:
        log_all_configs(logger)
    # setup dataset, model and dataloader
    with accelerator.main_process_first():
        dataset = ...
        model = ...
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        **config.DataLoaderConfig(),
    )
    # setup optimizer and lr scheduler
    optimizer = AdamW(
        params=model.parameters(), 
        **config.OptimizerConfig(),
    )
    num_steps_per_epoch = len(data_loader) // config.AcceleratorConfig().gradient_accumulation_steps
    num_epochs = config.TrainerConfig().epochs
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_steps_per_epoch * num_epochs,
    )
    # setup trainer and run
    trainer = ...
    trainer.run()

if __name__ == '__main__':
    launch(main, **config.LaunchConfig())
