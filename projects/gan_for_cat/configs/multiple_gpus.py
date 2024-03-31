from os import cpu_count
from pathlib import Path

from hurricane.config_base import ConfigBase
from hurricane.utils import set_cuda_visible_devices, get_config_name


# set_cuda_visible_devices(1)


config_name = get_config_name()
gradient_accumulation_interval = 8


class LaunchConfig(ConfigBase):
    num_processes = 4
    use_port = "8002"


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    logs = data / 'logs'
    dataset_path = data / 'afhq'
    checkpoints = data / 'checkpoints' / config_name
    tensor_boards = data / 'tensor_boards' / config_name
    peek_images = data / 'peek_images' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class LoggerConfig(ConfigBase):
    name = config_name
    level = 'INFO'
    logs_dir = PathConfig().logs


class GeneratorConfig(ConfigBase):
    z_dim = 1024
    hidden_dim = 512
    image_size = 256


class DiscriminatorConfig(ConfigBase):
    hidden_dim = 512
    image_size = 256


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_interval

    
class DataLoaderConfig(ConfigBase):
    batch_size = 8
    num_workers = cpu_count()
    shuffle = True


class TrainerConfig(ConfigBase):
    epochs = 5000
    d_loop_per_step = 3
    g_loop_per_step = 1
    
    log_interval = gradient_accumulation_interval
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_interval
    
    image_peek_folder_path = PathConfig().peek_images
    image_peek_interval = gradient_accumulation_interval
    
    checkpoint_folder_path = PathConfig().checkpoints
    checkpoint_interval = gradient_accumulation_interval * 1000
    checkpoint_seed = 42
    
    lr_scheduler_mode = 'per_step'


class DatasetConfig(ConfigBase):
    path = PathConfig().dataset_path
    image_size = 256


class GeneratorOptimizerConfig(ConfigBase):
    lr = 2e-4
    weight_decay = 1e-2


class DiscriminatorOptimizerConfig(ConfigBase):
    lr = 2e-4
    weight_decay = 1e-2
    