from os import cpu_count
from pathlib import Path

from accelerate import DataLoaderConfiguration

from hurricane.config_base import ConfigBase

config_name = 'default'
gradient_accumulation_steps = 1

class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints' / config_name
    tensor_boards = data / 'tensor_boards' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)

class LoggerConfig(ConfigBase):
    name = config_name
    level = 'INFO'
    logs_dir = PathConfig().logs

class SNGANConfig(ConfigBase):
    z_dim: int = 1024

class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_steps
    dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)
    
class DataLoaderConfig(ConfigBase):
    batch_size = 64
    num_workers = cpu_count()
    shuffle = True

class TrainerConfig(ConfigBase):
    epochs = 100
    seed = 42
    log_interval = gradient_accumulation_steps
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_steps