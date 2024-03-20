import path_setup

from os import cpu_count
import logging
from pathlib import Path

from hurricane.config_base import ConfigBase
from hurricane.utils import get_current_date_time


class PathConfig(ConfigBase):
    project = Path(__file__).parent
    data = project / 'data'
    cifar10_dataset = data / 'cifar10_dataset'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints'

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    epochs = 10
    ckpt_folder_path=PathConfig().checkpoints
    log_interval = 1


class OptimizerConfig(ConfigBase):
    lr = 1e-3


class DatasetConfig(ConfigBase):
    root=PathConfig().cifar10_dataset
    train=True
    download=True


class DataLoaderConfig(ConfigBase):
    batch_size = 512
    shuffle = True
    num_workers = cpu_count()


class LoggerConfig(ConfigBase):
    name = get_current_date_time()
    level = logging.INFO
    logs_dir = PathConfig().logs


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = 1


class CKPTConfig(ConfigBase):
    folder_path = PathConfig().checkpoints
