import path_setup

from torch.optim import AdamW
from accelerate import DeepSpeedPlugin

from hurricane.common_configs import *


gradient_accumulate_interval = 32


class PeekConfig(ConfigBase):
    prompts = [
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ]
    interval = gradient_accumulate_interval


class TrainingConfig(ConfigBase):
    epochs = 100
    lr = 3e-4
    batch_size = 4
    max_len = 512


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
    mixed_precision = 'fp16'
    split_batches = True
    deepspeed_plugin=DeepSpeedPlugin(
        gradient_accumulation_steps = gradient_accumulate_interval, 
        zero_stage = 3,
        offload_optimizer_device = 'cpu',
        # offload_param_device = 'cpu',
        zero3_init_flag = False,
    )
