import path_setup

from accelerate import DeepSpeedPlugin

from hurricane.common_configs import *


gradient_accumulate_interval = 8


class PeekConfig(ConfigBase):
    prompts = [
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ]
    interval = gradient_accumulate_interval * 4


class TrainingConfig(ConfigBase):
    epochs = 100
    lr = 5e-5
    batch_size_per_device = 8
    max_len = 512
    log_interval = gradient_accumulate_interval


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval


class CKPTConfig(ConfigBase):
    folder_path = Path(__file__).resolve().parent / 'checkpoints'
    