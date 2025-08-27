import shutil
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader 
from accelerate import Accelerator


class Checkpointer:
    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator
        self.register_state = accelerator.register_for_checkpointing
        self.registered_actions = []


    def save_state(self, ckpt_path: Path) -> Path:
        if ckpt_path.exists() and self.accelerator.is_main_process:
            shutil.rmtree(ckpt_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(ckpt_path))
        return ckpt_path


    def load_state(self, ckpt_dir: Path) -> None:
        self.accelerator.load_state(str(ckpt_dir))
    

    def reprepare_data_loader(self, data_loader: DataLoader, seed: int = 42) -> DataLoader:
        torch.manual_seed(seed)
        self.accelerator.dataloader_config.use_seedable_sampler = True
        return self.accelerator.prepare(data_loader)
    

    def register_action(self, func: Callable, *args, **kwargs) -> None:
        self.registered_actions.append((func, args, kwargs))
    

    def excute_actions(self) -> None:
        for action in self.registered_actions:
            func, args, kwargs = action
            func(*args, **kwargs)
