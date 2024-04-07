import _path_setup

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricane.utils import Logger, launch, log_all_configs

from configs.default import (
    LoggerConfig,
    AcceleratorConfig,
    DataLoaderConfig,
    OptimizerConfig,
    TrainerConfig,
    LaunchConfig,
)


def main():
    logger = Logger(**LoggerConfig())
    accelerator = Accelerator(**AcceleratorConfig())
    if accelerator.is_main_process:
        log_all_configs(logger)
    with accelerator.main_process_first():
        dataset = ...
        model = ...
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        **DataLoaderConfig(),
    )
    optimizer = AdamW(
        params=model.parameters(), 
        **OptimizerConfig(),
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=(len(data_loader) // AcceleratorConfig().gradient_accumulation_steps) * TrainerConfig().epochs,
    )
    trainer = ...
    trainer.run()

if __name__ == '__main__':
    launch(main, **LaunchConfig())
