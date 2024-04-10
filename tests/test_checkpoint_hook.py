import _path_setup  # noqa: F401

import shutil
from pathlib import Path


import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers import Trainer
from hurricane.hooks import CheckpointHook


temp_folder_path = Path(__file__).parents[1] / '_temp_checkpoints'



class _TestTrainer(Trainer):
    def __init__(self):
        model = nn.Linear(1, 1)
        super().__init__(
            models=[model],
            optimizers=[AdamW(model.parameters(), lr=1e-3)],
            data_loaders=[DataLoader(range(10), batch_size=1, shuffle=True)],
            accelerator=Accelerator(),
            epochs=2,
        )
        self.hooks = [CheckpointHook(self, folder_path=temp_folder_path, interval=5)]
        self.iterated_results = []
    
    
    def training_step(self) -> torch.Tensor:
        batch = self.accelerator.gather(
            self.ctx.batches[0]
        ).sort()[0]
        self.iterated_results.append(batch)
        return torch.tensor(0.0)
    

def test_checkpoint_hook():
    # set up test folder
    if temp_folder_path.exists():
        shutil.rmtree(temp_folder_path)
    temp_folder_path.mkdir(parents=True)
    
    trainer = _TestTrainer()
    trainer.run()
    assert len(trainer.iterated_results) == 20, "Not all batches are iterated."
    original_results = trainer.iterated_results.copy()
    del trainer
    
    # remove all but the 5th and the 15th
    ckpt_dirs = [d for d in temp_folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_step_')]
    for ckpt_dir in ckpt_dirs:
        if ckpt_dir.name not in ['ckpt_step_5', 'ckpt_step_15']:
            shutil.rmtree(ckpt_dir)
    
    # test reproducibility on the 2nd epoch
    trainer = _TestTrainer()
    trainer.run()
    assert len(trainer.iterated_results) == 5, "Continued number of batches is not correct."
    new_results = trainer.iterated_results.copy()
    assert new_results == original_results[15:], "Continued batches do not match the original."
    del trainer

    # remove all but the 5th
    ckpt_dirs = [d for d in temp_folder_path.iterdir() if d.is_dir() and d.name.startswith('ckpt_step_')]
    for ckpt_dir in ckpt_dirs:
        if ckpt_dir.name not in ['ckpt_step_5']:
            shutil.rmtree(ckpt_dir)
    
    # test reproducibility on the 1st epoch
    trainer = _TestTrainer()
    trainer.run()
    assert len(trainer.iterated_results) == 15, "Continued number of batches is not correct."
    new_results = trainer.iterated_results.copy()
    assert new_results == original_results[5:], "Continued batches do not match the original."
    del trainer
    
    # clean up
    shutil.rmtree(temp_folder_path)
