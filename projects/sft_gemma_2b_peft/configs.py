import path_setup

from torch.optim import AdamW
from peft import LoraConfig, TaskType

from hurricane.common_configs import *


gradient_accumulate_interval = 8


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
    batch_size = 16
    max_len = 512


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = gradient_accumulate_interval
    mixed_precision = 'fp16'
    split_batches = True


class PEFTConfig(ConfigBase):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )


class CKPTConfig(ConfigBase):
    interval = 1
    folder_path = Path(__file__).resolve().parent / 'checkpoints'