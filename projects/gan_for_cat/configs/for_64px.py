from os import cpu_count
from pathlib import Path

from hurricane.utils import ConfigBase, set_cuda_visible_devices, get_config_name


set_cuda_visible_devices(2, 3)

image_size = 64
epochs = 5000
batch_size = 128
lr = 2e-4
peek_interval = 100
ckpt_interval = 1000
gradient_accumulation_interval = 1

config_name = get_config_name()


class LaunchConfig(ConfigBase):
    num_processes = 2
    use_port = "8001"


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
    hidden_dim = 128
    image_size = image_size


class DiscriminatorConfig(ConfigBase):
    hidden_dim = 192
    image_size = image_size


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_interval

    
class DataLoaderConfig(ConfigBase):
    batch_size = batch_size
    num_workers = cpu_count()
    shuffle = True


class TrainerConfig(ConfigBase):
    epochs = epochs
    gp_lambda = 10
    d_loop_per_step = 3
    g_loop_per_step = 1
    
    log_interval = gradient_accumulation_interval
    
    tensor_board_folder_path = PathConfig().tensor_boards
    tensor_board_interval = gradient_accumulation_interval
    
    image_peek_folder_path = PathConfig().peek_images
    image_peek_interval = gradient_accumulation_interval * peek_interval
    
    checkpoint_folder_path = PathConfig().checkpoints
    checkpoint_interval = gradient_accumulation_interval * ckpt_interval
    checkpoint_seed = 42
    
    lr_scheduler_mode = 'per_step'


class DatasetConfig(ConfigBase):
    path = PathConfig().dataset_path
    image_size = image_size


class GeneratorOptimizerConfig(ConfigBase):
    lr = lr
    # weight_decay = 1e-2
    betas=(0.5, 0.9)


class DiscriminatorOptimizerConfig(ConfigBase):
    lr = lr
    # weight_decay = 1e-2
    betas=(0.5, 0.9)
    