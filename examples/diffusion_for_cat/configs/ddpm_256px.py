from os import cpu_count
import logging
from pathlib import Path

from hurricane.utils import ConfigBase, get_file_name


num_diffusion_steps = 1000
image_size = 256
num_epochs = 2000
batch_size = 8
lr = 1e-4
gradient_accumulation_interval = 4
ckpt_interval = 1000
image_peek_interval = 500

config_name = get_file_name()


class DDPMNoiseSchedulerConfig(ConfigBase):
    beta_start = 1e-4
    beta_end = 2e-2
    num_steps = num_diffusion_steps


class UNetConfig(ConfigBase):
    image_size = image_size
    layers_per_block = 2
    block_out_channels = (64, 64, 128, 256, 512, 512)


class LaunchConfig(ConfigBase):
    num_processes = 2
    use_port = "8000"


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    dataset = data / 'afhq'
    logs = data / 'logs'
    peek_images = data / 'peek_images' / config_name
    checkpoints = data / 'checkpoints' / config_name
    tensor_boards = data / 'tensor_boards' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    num_epochs = num_epochs
    
    log_interval = gradient_accumulation_interval
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_interval
    
    image_peek_folder_path = PathConfig().peek_images
    image_peek_interval = gradient_accumulation_interval * image_peek_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulation_interval * ckpt_interval
    ckpt_seed = 42
    
    lr_scheduler_mode = 'per_step'


class OptimizerConfig(ConfigBase):
    lr = lr
    # betas=(0.5, 0.9)


class DatasetConfig(ConfigBase):
    path = PathConfig().dataset
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
