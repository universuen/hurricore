# import path_setup

# from accelerate import DeepSpeedPlugin

# from hurricane.common_configs import *


# 


# class PeekConfig(ConfigBase):
#     prompts = [
#         '如何看待明天下雨？',
#         '为什么太阳比地球大？',
#         '你如何看待近期的股市？',
#     ]
#     interval = gradient_accumulate_interval * 10


# class TrainingConfig(ConfigBase):
#     epochs = 100
#     lr = 5e-5
#     batch_size_per_device = 12
#     max_len = 512
#     log_interval = gradient_accumulate_interval


# class AcceleratorConfig(ConfigBase):
#     gradient_accumulation_steps = gradient_accumulate_interval


# class CKPTConfig(ConfigBase):
#     folder_path = Path(__file__).resolve().parent / 'checkpoints'


import path_setup

from os import cpu_count
import logging
from pathlib import Path

from hurricane.config_base import ConfigBase


gradient_accumulate_interval = 8


class PathConfig(ConfigBase):
    project = Path(__file__).parent
    data = project / 'data'
    cifar10_dataset = data / 'cifar10_dataset'
    logs = data / 'logs'
    checkpoints = data / 'checkpoints'

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class TrainerConfig(ConfigBase):
    epochs = 100
    ckpt_folder_path=PathConfig().checkpoints
    log_interval = 1
    peek_prompts=[
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ],
    peek_interval=gradient_accumulate_interval * 10
    log_interval=gradient_accumulate_interval
    ckpt_folder_path=PathConfig().checkpoints


class OptimizerConfig(ConfigBase):
    lr = 5e-5


class DataLoaderConfig(ConfigBase):
    batch_size = 32
    shuffle = True
    num_workers = cpu_count()


class HFLLMITCollatorConfig(ConfigBase):
    max_len = 512


class LoggerConfig(ConfigBase):
    level = logging.INFO
    logs_dir = PathConfig().logs


class AcceleratorConfig(ConfigBase):
    split_batches = True
    gradient_accumulation_steps = 1


class CKPTConfig(ConfigBase):
    folder_path = PathConfig().checkpoints
