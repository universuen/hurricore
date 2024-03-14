from __future__ import annotations

from types import MethodType, FunctionType

from accelerate import notebook_launcher


def enable_grad(func: MethodType | FunctionType) -> MethodType | FunctionType:
    return func.__closure__[1].cell_contents


launch_for_parallel_training = notebook_launcher

