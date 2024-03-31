from os import cpu_count
import logging
from pathlib import Path

from hurricane.config_base import ConfigBase
from hurricane.utils import get_config_name


config_name = get_config_name()
gradient_accumulate_interval = 1


class LaunchConfig(ConfigBase):
    num_processes = 2
    use_port = "8002"


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints' / config_name
    tensor_boards = data / 'tensor_boards' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    epochs = ...
    
    log_interval = gradient_accumulate_interval
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulate_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulate_interval * 10
    ckpt_seed = 42
    


class OptimizerConfig(ConfigBase):
    ...


class DatasetConfig(ConfigBase):
    ...


class DataLoaderConfig(ConfigBase):
    batch_size = ...
    shuffle = True
    num_workers = cpu_count()


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs
    

class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
