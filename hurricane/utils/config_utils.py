from __future__ import annotations

import os
import sys
import inspect
import requests
import argparse
from pathlib import Path
from types import ModuleType
from importlib import util, import_module


def get_file_name() -> str:
    caller_frame_record = inspect.stack()[1]
    module_path = caller_frame_record.filename
    return Path(module_path).stem


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
