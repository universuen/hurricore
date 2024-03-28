import _path_setup

from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricane.trainers import Trainer
from hurricane.hooks import CheckpointHook
from hurricane.utils import launch


memory = []

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
            ),
        ]
        
    def compute_loss(self):
        memory.append(
            (self.ctx.epoch, self.ctx.batch_idx, self.ctx.batch)
        )
        output = self.model(self.ctx.batch)
        return output.sum()

def setup():
    model = nn.Linear(3, 3)

    global memory
    memory = []
    data_loader = DataLoader(torch.arange(0, 512 * 3).reshape(512, 3).float(), batch_size=32, shuffle=True)
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
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        remove_ckpt_folder(ckpt_folder_path)
    trainer.accelerator.wait_for_everyone()


def get_optimizer_state(optimizer):
    values = []
    for state in optimizer.state_dict()['state'].values():
        for value in state.values():
            if value.dim() == 0:
                value = value.unsqueeze(0)
            else:
                value = value.flatten()
            values.append(value.cpu())
    return torch.cat(values)


def test_optimizer_state_is_identical():
    trainer = setup()
    # run training
    trainer.run()
    # record updated optimizer state
    updated_state = get_optimizer_state(trainer.originals.optimizer)
    # reload the model
    del trainer
    trainer = setup()
    # load the checkpoint
    trainer.get_hook(CheckpointHook).on_training_start()
    # record checkpoint optimizer state
    checkpoint_state = get_optimizer_state(trainer.originals.optimizer)
    # should be identical
    assert torch.allclose(updated_state, checkpoint_state)
    # clean up
    ckpt_folder_path = trainer.get_hook(CheckpointHook).folder_path
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        remove_ckpt_folder(ckpt_folder_path)
    trainer.accelerator.wait_for_everyone()

def test_dataloader_is_resumed_correctly():
    trainer = setup()
    # run training
    trainer.run()
    # record memory after checkpoint saving
    memory_in_original_process = deepcopy(memory[1:])
    # reload trainer
    del trainer
    trainer = setup()
    # load the checkpoint at step 1, this requires removing latter checkpoints
    ckpt_folder_path = trainer.get_hook(CheckpointHook).folder_path
    
    if trainer.accelerator.is_main_process:
        for item in ckpt_folder_path.iterdir():
            if item.is_dir() and item.name != 'ckpt_step_1':
                remove_ckpt_folder(item)
    trainer.accelerator.wait_for_everyone()
    # run training
    trainer.run()
    # record memory after resuming
    memory_in_resumed_process = deepcopy(memory)
    if trainer.accelerator.is_main_process:
        remove_ckpt_folder(ckpt_folder_path)
    # should be identical
    results_a = [i[-1] for i in memory_in_original_process]
    results_b = [i[-1] for i in memory_in_resumed_process]
    
    for a in results_a:
        has_same_element = False
        for b in results_b:
            if torch.equal(a, b):
                has_same_element = True
                break
        assert has_same_element, 'Resumed dataloader is not correct.'
    trainer.accelerator.wait_for_everyone()


# launch all func that start with test_
for func_name in dir():
    if func_name.startswith('test_'):
        func = locals()[func_name]
        launch(func, num_processes=2, use_port='8000')
