import logging
from pathlib import Path

from hurricane.utils import ConfigBase, get_file_name


num_epochs = 2
batch_size = 256
lr = 1e-3
gradient_accumulation_interval = 2
config_name = get_file_name()


class LaunchConfig(ConfigBase):
    num_processes = 2
    use_port = "8009"


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
    num_epochs = num_epochs
    
    log_interval = gradient_accumulation_interval
    
    tensor_board_folder_path=PathConfig().tensor_boards
    tensor_board_interval=gradient_accumulation_interval
    
    ckpt_folder_path=PathConfig().checkpoints
    ckpt_interval = gradient_accumulation_interval * 10
    ckpt_seed = 42
    


class OptimizerConfig(ConfigBase):
    lr = lr


class DatasetConfig(ConfigBase):
    root=PathConfig().cifar10_dataset
    train=True
    download=True


class DataLoaderConfig(ConfigBase):
    batch_size = batch_size
    shuffle = True
    num_workers = 1  # cpu_count()


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs
    

class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_interval
