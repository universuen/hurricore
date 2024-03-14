from __future__ import annotations

from types import MethodType, FunctionType

from accelerate import notebook_launcher


launch_for_parallel_training = notebook_launcher

def get_list_mean(list_: list[int | float]) -> float:
    return sum(list_) / (len(list_) + 1e-6)
