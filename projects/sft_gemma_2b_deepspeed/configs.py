import path_setup

from accelerate import DeepSpeedPlugin

from hurricane.common_configs import *


gradient_accumulation_steps = 32


class PeekConfig(ConfigBase):
    prompts = [
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ]
    interval = gradient_accumulation_steps * 10


class TrainingConfig(ConfigBase):
    epochs = 100
    lr = 5e-5
    batch_size_per_device = 1
    max_len = 512
    log_interval = gradient_accumulation_steps


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_steps
    deepspeed_plugin=DeepSpeedPlugin(
        gradient_accumulation_steps = gradient_accumulation_steps, 
        zero_stage = 2,
        offload_optimizer_device = 'cpu',
        zero3_init_flag = False,
    )


class CKPTConfig(ConfigBase):
    folder_path = Path(__file__).resolve().parent / 'checkpoints'
    