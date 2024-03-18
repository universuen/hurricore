import path_setup

from accelerate import DeepSpeedPlugin

from hurricane.common_configs import *


gradient_accumulation_steps = 16


class PeekConfig(ConfigBase):
    prompts = [
        '如何看待明天下雨？',
        '为什么太阳比地球大？',
        '你如何看待近期的股市？',
    ]
    interval = gradient_accumulation_steps


class TrainingConfig(ConfigBase):
    epochs = 100
    lr = 3e-4
    batch_size = 4
    max_len = 512
    log_interval = gradient_accumulation_steps


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulation_steps
    split_batches = True
    deepspeed_plugin=DeepSpeedPlugin(
        gradient_accumulation_steps = gradient_accumulation_steps, 
        zero_stage = 2,
        offload_optimizer_device = 'cpu',
        zero3_init_flag = False,
    )


class CKPTConfig(ConfigBase):
    folder_path = Path(__file__).resolve().parent / 'checkpoints'
    