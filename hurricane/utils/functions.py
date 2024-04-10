from __future__ import annotations

import os
import sys
import inspect
import requests
import tempfile
from datetime import datetime
from pathlib import Path
from logging import Logger as LoggerType
from typing import Iterable
from importlib import util, import_module

import torch
from accelerate import notebook_launcher

from hurricane.utils import ConfigBase


launch = notebook_launcher


def log_all_configs(logger: LoggerType) -> None:
    for subclass in ConfigBase.__subclasses__():
        logger.info(subclass())


def get_list_mean(list_: list[int | float]) -> float:
    return sum(list_) / (len(list_) + 1e-6)


def find_start_and_end_index(a: torch.Tensor, b: torch.Tensor) -> int:
        for i in range(len(a) - len(b) + 1):
            if torch.all(a[i:i + len(b)] == b):
                return i, i + len(b)
        return -1, -1


def get_file_name() -> str:
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


def auto_name(iterable: Iterable) -> list[str]:
    names = []
    names_cnt = {}
    for i in iterable:
        name = i.__class__.__name__
        if name in names_cnt:
            names_cnt[name] += 1
            name = f'{name}_{names_cnt[name]}'
        else:
            names_cnt[name] = 0
        names.append(name)
    return names


def format_parameters(num_params):
    if num_params >= 1e9: 
        return f'{num_params / 1e9:.2f} B'
    elif num_params >= 1e6: 
        return f'{num_params / 1e6:.2f} M'
    elif num_params >= 1e3: 
        return f'{num_params / 1e3:.2f} K'
    else: 
        return str(num_params)


def get_total_parameters(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    return format_parameters(total_params)


def get_trainable_parameters(model: torch.nn.Module) -> str:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format_parameters(trainable_params)


def set_cuda_visible_devices(*device_indices: tuple[int]) -> None:
    assert all(isinstance(i, int) for i in device_indices), 'device_indices must be integers'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_indices))


def import_config(file_or_url_path: str):
    path = file_or_url_path
    assert isinstance(path, str), "path must be a string"
    temp_file_path = None
    if path.startswith("http"):
        # Handle URL
        response = requests.get(path)
        if response.status_code != 200:
            raise ImportError(f"Cannot download the module from {path}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode='w+') as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(response.text)
            module_path = tmp_file.name
            module_name = Path(tmp_file.name).stem
        spec = util.spec_from_file_location(module_name, module_path)
    elif  "/" in path or "\\" in path:
        # Handle filesystem path
        path = Path(path)
        module_path = str(path)
        module_name = Path(path).stem
        spec = util.spec_from_file_location(module_name, module_path)
    else:
        # Handle module import string
        try:
            return import_module(path)
        except ImportError as e:
            raise ImportError(f"Cannot import module using import string {path}") from e

    if spec is None:
        raise ImportError(f"Cannot import module from {path}")
    
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if temp_file_path is not None:
        print(temp_file_path)
        os.remove(temp_file_path)
    
    return module

