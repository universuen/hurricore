from os import cpu_count
import logging
from pathlib import Path

from accelerate import DeepSpeedPlugin, DataLoaderConfiguration

from hurricane.config_base import ConfigBase
from hurricane.utils import get_config_name

model_name = "google/gemma-2b"
config_name = get_config_name()
gradient_accumulate_interval = 32


class PathConfig(ConfigBase):
    project = Path(__file__).parents[1]
    data = project / 'data'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints' / config_name
    tensorboards = data / 'tensorboards' / config_name

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    epochs = 10
    
    log_interval = gradient_accumulate_interval
    
    peek_prompts = [
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ]
    peek_interval=gradient_accumulate_interval * 10

    tensorboard_folder_path = PathConfig().tensorboards
    tensorboard_interval = gradient_accumulate_interval
    
    ckpt_folder_path = PathConfig().checkpoints
    ckpt_interval = gradient_accumulate_interval * 1000


class OptimizerConfig(ConfigBase):
    lr = 5e-5


class DataLoaderConfig(ConfigBase):
    batch_size = 1
    shuffle = True
    num_workers = cpu_count()


class CollatorConfig(ConfigBase):
    max_len = 512


class LoggerConfig(ConfigBase):
    name = config_name
    level = logging.INFO
    logs_dir = PathConfig().logs


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
    deepspeed_plugin=DeepSpeedPlugin(
        gradient_accumulation_steps = gradient_accumulation_steps, 
        zero_stage = 2,
        offload_optimizer_device = 'cpu',
        zero3_init_flag = False,
    )
    dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)