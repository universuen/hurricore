from os import cpu_count
import logging
from pathlib import Path

from hurricane.utils import ConfigBase, get_file_name, set_cuda_visible_devices


set_cuda_visible_devices(0, 3)
image_size = 256
num_epochs = 2000
batch_size = 8
lr = 1e-4
gradient_accumulation_interval = 2
log_interval = 10
tensor_board_interval = 10
ckpt_interval = 1000
img_peek_interval = 500
config_name = get_file_name()


class UNetConfig(ConfigBase):
    image_size = image_size
    layers_per_block = 2
    block_out_channels = (128, 128, 256, 256, 512, 512)


class LaunchConfig(ConfigBase):
    num_processes = 2
    use_port = "8001"


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    dataset = data / 'afhq'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints' / config_name
    tensor_boards = data / 'tensor_boards' / config_name
    img_peek = data / 'img_peek' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class FlowTrainerConfig(ConfigBase):
    num_epochs = num_epochs
    
    img_peek_folder_path = PathConfig().img_peek
    img_peek_interval = gradient_accumulation_interval * img_peek_interval
    
    log_interval = gradient_accumulation_interval * log_interval
    
    lr_scheduler_mode = 'per_step'
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_interval * tensor_board_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulation_interval * ckpt_interval
    ckpt_seed = 42
    

class OptimizerConfig(ConfigBase):
    lr = lr


class TrainingNoiseCatDatasetConfig(ConfigBase):
    path = PathConfig().dataset
    seed = 0
    image_size = image_size


class ValidationNoiseCatDatasetConfig(ConfigBase):
    path = PathConfig().dataset
    seed = 1
    image_size = image_size


class DataLoaderConfig(ConfigBase):
    batch_size = batch_size
    shuffle = True
    num_workers = cpu_count()


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs
    

class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_interval
    mixed_precision = 'fp16'
