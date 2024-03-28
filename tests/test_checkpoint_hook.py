import _path_setup

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator


from hurricane.trainers import Trainer
from hurricane.hooks import CheckpointHook


class DummyTrainer(Trainer):
    def __init__(self, model, data_loader, optimizer, accelerator):
        super().__init__(model, data_loader, optimizer, accelerator, 1)
        ckpt_folder_path = Path(__file__).parent / 'test_ckpt'
        ckpt_folder_path.mkdir(exist_ok=True)
        self.hooks = [
            CheckpointHook(
                trainer=self,
                folder_path=ckpt_folder_path,
                interval=1,
            )
        ]
        
    def compute_loss(self):
        output = self.model(self.ctx.batch)
        return output.sum()

def setup():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )
    data_loader = DataLoader(torch.rand(30, 10), batch_size=10, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)
    accelerator = Accelerator()
    trainer = DummyTrainer(model, data_loader, optimizer, accelerator)
    return trainer


def remove_ckpt_folder(ckpt_folder_path: Path):
    for item in ckpt_folder_path.iterdir():
        if item.is_dir():
            remove_ckpt_folder(item)
        else:
            item.unlink()
    ckpt_folder_path.rmdir()


def test_model_parameters_are_identical():
    trainer = setup()
    # record the model parameters
    origianl_parameters = torch.cat(
        [
            param.data.clone().flatten()
            for param in trainer.originals.model.parameters()
        ]
    )
    # run training
    trainer.run()
    # record updated model parameters
    updated_parameters = torch.cat(
        [
            param.data.clone().flatten()
            for param in trainer.originals.model.parameters()
        ]
    )
    # should be different
    assert not torch.allclose(origianl_parameters, updated_parameters)

    # reload the model
    del trainer
    trainer = setup()
    # record reloaded model parameters
    reloaded_parameters = torch.cat(
        [
            param.data.clone().flatten()
            for param in trainer.originals.model.parameters()
        ]
    )
    # should be different
    assert not torch.allclose(updated_parameters, reloaded_parameters)
    # load the checkpoint
    trainer.get_hook(CheckpointHook).on_training_start()
    # record checkpoint model parameters
    checkpoint_parameters = torch.cat(
        [
            param.data.clone().flatten()
            for param in trainer.originals.model.parameters()
        ]
    )
    # should be identical
    assert torch.allclose(updated_parameters, checkpoint_parameters)
    
    # clean up
    ckpt_folder_path = trainer.get_hook(CheckpointHook).folder_path
    remove_ckpt_folder(ckpt_folder_path)
