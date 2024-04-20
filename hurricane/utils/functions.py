from __future__ import annotations

import os
import sys
import inspect
import requests
import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable
from types import ModuleType
from importlib import util, import_module

import torch
from accelerate import notebook_launcher


launch = notebook_launcher


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


def import_config(path: str, accept_cmd_args: bool = True) -> ModuleType:
    if accept_cmd_args:
        help_msg = (
            "CONFIG can be any of the following formats:\n"
            "- python_module.config\n"
            "- local/file/path/to/config.py\n"
            "- https://url/to/config.py"
        )
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-c', '--config', type=str, default=path, help=help_msg)
        args = parser.parse_args()
        path = args.config
    
    from hurricane.hooks import LoggerHook
    LoggerHook.msg_queue.append(
        ('info', f'Imported config from {path}')
    )
    
    assert isinstance(path, str), "path must be a string"
    
    temp_file_path = None
    if path.startswith("http"):
        # handle URL
        response = requests.get(path)
        if response.status_code != 200:
            raise ImportError(f"Cannot download the module from {path}")
        with open(Path(__file__).parents[2] / '_temp_config_from_url.py', 'w') as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(response.text)
            module_path = tmp_file.name
            module_name = Path(tmp_file.name).stem
        spec = util.spec_from_file_location(module_name, module_path)
    elif  "/" in path or "\\" in path:
        # handle filesystem path
        path = Path(path)
        module_path = str(path)
        module_name = Path(path).stem
        spec = util.spec_from_file_location(module_name, module_path)
    else:
        # handle module import string
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
        os.remove(temp_file_path)
    
    return module


def get_params_details_table(*models: tuple[torch.nn.Module]) -> str:
        table_header = f"\t{'Model':<30} | {'Total Params':>20} | {'Trainable Params':>20}"
        table_divider = '\t' + '-' * len(table_header)
        table_rows = [table_header, table_divider]
        for name, model in zip(auto_name(models), models):
            total_params = get_total_parameters(model)
            trainable_params = get_trainable_parameters(model)
            row = f"\t{name:<30} | {total_params:>20} | {trainable_params:>20}"
            table_rows.append(row)
        full_table = '\n'.join(table_rows)
        return f'\n{full_table}\n'
