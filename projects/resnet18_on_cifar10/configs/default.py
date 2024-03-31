from os import cpu_count
import logging
from pathlib import Path

from hurricane.config_base import ConfigBase
from hurricane.utils import get_config_name, set_cuda_visible_devices

set_cuda_visible_devices(4, 5, 6, 7)


config_name = get_config_name()
gradient_accumulate_interval = 1


class LaunchConfig(ConfigBase):
    num_processes = 4
    use_port = "8000"


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    cifar10_dataset = data / 'cifar10_dataset'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints' / config_name
    tensor_boards = data / 'tensor_boards' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    epochs = 2
    
    log_interval = gradient_accumulate_interval
    
    tensor_board_folder_path=PathConfig().tensor_boards
    tensor_board_interval=gradient_accumulate_interval
    
    ckpt_folder_path=PathConfig().checkpoints
    ckpt_interval = gradient_accumulate_interval * 10
    ckpt_seed = 42
    


class OptimizerConfig(ConfigBase):
    lr = 1e-3


class DatasetConfig(ConfigBase):
    root=PathConfig().cifar10_dataset
    train=True
    download=True


class DataLoaderConfig(ConfigBase):
    batch_size = 8
    shuffle = True
    num_workers = 1  # cpu_count()


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs
    

class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
