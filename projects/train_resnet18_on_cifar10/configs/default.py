from os import cpu_count
import logging
from pathlib import Path
from accelerate import DataLoaderConfiguration

from hurricane.config_base import ConfigBase
from hurricane.utils import get_config_name


config_name = get_config_name()
gradient_accumulate_interval = 4


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    cifar10_dataset = data / 'cifar10_dataset'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints' / config_name
    tensorboards = data / 'tensorboards' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    epochs = 10
    seed = 42
    
    log_interval = gradient_accumulate_interval
    
    tensorboard_folder_path=PathConfig().tensorboards
    tensorboard_interval=gradient_accumulate_interval
    
    ckpt_folder_path=PathConfig().checkpoints
    ckpt_interval = gradient_accumulate_interval * 100


class OptimizerConfig(ConfigBase):
    lr = 1e-3


class DatasetConfig(ConfigBase):
    root=PathConfig().cifar10_dataset
    train=True
    download=True


class DataLoaderConfig(ConfigBase):
    batch_size = 512
    shuffle = True
    num_workers = 1  # cpu_count()


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
    dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)    
