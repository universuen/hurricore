from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path
from logging import Logger

import torch
from accelerate import notebook_launcher

from hurricane.config_base import ConfigBase


launch = notebook_launcher


def log_all_configs(logger: Logger) -> None:
    for subclass in ConfigBase.__subclasses__():
        logger.info(subclass())


def get_list_mean(list_: list[int | float]) -> float:
    return sum(list_) / (len(list_) + 1e-6)


def find_start_and_end_index(a: torch.Tensor, b: torch.Tensor) -> int:
        for i in range(len(a) - len(b) + 1):
            if torch.all(a[i:i + len(b)] == b):
                return i, i + len(b)
        return -1, -1


def get_config_name() -> str:
    caller_frame_record = inspect.stack()[1]
    module_path = caller_frame_record.filename
    return Path(module_path).stem


def is_deepspeed_zero3(accelerator) -> bool:
    if accelerator.state.deepspeed_plugin is not None \
    and accelerator.state.deepspeed_plugin.zero_stage == 3:
        return True
    return False


def get_time_stamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")
