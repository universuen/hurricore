from os import cpu_count
import logging
from pathlib import Path
from accelerate import DataLoaderConfiguration

from hurricane.utils import ConfigBase, get_file_name


num_epochs = 10
batch_size = 12
lr = 5e-5
model_name = "facebook/opt-350m"
gradient_accumulation_interval = 8

config_name = get_file_name()


class LaunchConfig(ConfigBase):
    num_processes = 4
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
    num_epochs = num_epochs
    
    log_interval = gradient_accumulation_interval
    
    peek_prompts = [
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ]
    peek_interval=gradient_accumulation_interval * 10

    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulation_interval * 1000
    ckpt_seed = 42
    

class OptimizerConfig(ConfigBase):
    lr = lr


class DataLoaderConfig(ConfigBase):
    batch_size = batch_size
    shuffle = True
    num_workers = cpu_count()


class CollatorConfig(ConfigBase):
    max_len = 512


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_interval
    dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)
