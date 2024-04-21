from os import cpu_count
import logging
from pathlib import Path

from hurricane.utils import ConfigBase, get_file_name, set_cuda_visible_devices

set_cuda_visible_devices(2, 3)

image_size = 128
num_epochs = 1000
batch_size = 16
lr = 1e-4
gradient_accumulation_interval = 1
ckpt_interval = 3000
img_peek_interval = 1000

config_name = get_file_name()


class UNetConfig(ConfigBase):
    image_size = image_size
    layers_per_block = 2
    block_out_channels = (128, 128, 256, 256, 512, 512)


class LaunchConfig(ConfigBase):
    num_processes = 2
    use_port = "8000"


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    training_dataset = data / 'afhq' / 'train'
    validation_dataset = data / 'afhq' / 'val'
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
    
    log_interval = gradient_accumulation_interval
    
    lr_scheduler_mode = 'per_step'
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulation_interval * ckpt_interval
    ckpt_seed = 42
    

class OptimizerConfig(ConfigBase):
    lr = lr


class TrainingCatDogDatasetConfig(ConfigBase):
    path = PathConfig().training_dataset
    image_size = image_size


class ValidationCatDogDatasetConfig(ConfigBase):
    path = PathConfig().validation_dataset
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
