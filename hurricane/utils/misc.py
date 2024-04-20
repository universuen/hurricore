from __future__ import annotations

from datetime import datetime
from typing import Iterable

import torch
from accelerate import notebook_launcher


launch = notebook_launcher


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
