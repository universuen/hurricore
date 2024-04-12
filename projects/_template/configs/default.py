from os import cpu_count
import logging
from pathlib import Path

from hurricane.utils import ConfigBase, get_file_name


num_epochs = 100
batch_size = 64
lr = 5e-5
gradient_accumulate_interval = 1
ckpt_interval = 10

config_name = get_file_name()


class LaunchConfig(ConfigBase):
    num_processes = 8
    use_port = "8000"


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
    num_epochs = num_epochs
    
    log_interval = gradient_accumulate_interval
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulate_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulate_interval * ckpt_interval
    ckpt_seed = 42
    

class OptimizerConfig(ConfigBase):
    lr = lr


class DatasetConfig(ConfigBase):
    ...


class DataLoaderConfig(ConfigBase):
    batch_size = batch_size
    shuffle = True
    num_workers = cpu_count()


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs
    

class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
