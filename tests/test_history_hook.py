import shutil
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricore.trainers import HurricoreTrainer
from hurricore.hooks import HistoryHook, CheckpointHook
from hurricore.utils import HistoryRecorder

temp_checkpoint_folder_path = Path(__file__).parents[1] / '_temp_checkpoints'
temp_history_folder_path = Path(__file__).parents[1] / '_temp_histories'


class _TestTrainer(HurricoreTrainer):
    def __init__(self):
        model = nn.Linear(1, 1)
        super().__init__(
            models=[model],
            optimizers=[AdamW(model.parameters(), lr=1e-3)],
            data_loaders=[DataLoader(range(10), batch_size=1, shuffle=True)],
            accelerator=Accelerator(),
            num_epochs=2,
            hooks=[
                CheckpointHook(self, temp_checkpoint_folder_path, interval=5),
                HistoryHook(self, temp_history_folder_path),
            ]
        )
        self.has_crashed = False

    def step(self, batch) -> torch.Tensor:
        history_hook = self.hooks[-1]
        # history_hook.msg_queue.append(
        #     ('record', {'tag': 'my_value', 'value': self.training_context.global_step, 'global_step': self.training_context.global_step}) 
        # )
        history_hook.recoder.record('my_value', self.training_context.global_step, self.training_context.global_step)
        if self.training_context.global_step == 17 and not self.has_crashed:
            self.has_crashed = True
            raise ValueError('Crash!')
        return torch.randn(1)


def test_history_hook():
    if temp_checkpoint_folder_path.exists():
        shutil.rmtree(temp_checkpoint_folder_path)
    if temp_history_folder_path.exists():
        shutil.rmtree(temp_history_folder_path)
    temp_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
    temp_history_folder_path.mkdir(parents=True, exist_ok=True)

    trainer = _TestTrainer()
    try:
        trainer.run()
    except ValueError:
        pass
    histories = HistoryRecorder.load(temp_history_folder_path)
    assert histories['my_value'] == [
        {'global_step': 0, 'value': 0},
        {'global_step': 1, 'value': 1},
        {'global_step': 2, 'value': 2},
        {'global_step': 3, 'value': 3},
        {'global_step': 4, 'value': 4},
        {'global_step': 5, 'value': 5},
        {'global_step': 6, 'value': 6},
        {'global_step': 7, 'value': 7},
        {'global_step': 8, 'value': 8},
        {'global_step': 9, 'value': 9},
        {'global_step': 10, 'value': 10},
        {'global_step': 11, 'value': 11},
        {'global_step': 12, 'value': 12},
        {'global_step': 13, 'value': 13},
        {'global_step': 14, 'value': 14},
        {'global_step': 15, 'value': 15},
        {'global_step': 16, 'value': 16},
        {'global_step': 17, 'value': 17},
    ]
    trainer.run()
    histories = HistoryRecorder.load(temp_history_folder_path)
    for idx, record in enumerate(histories['my_value']):
        assert record['global_step'] == idx, f"Global step is not correct. {record['global_step']} != {idx}"
        assert record['value'] == idx, f"Value is not correct. {record['value']} != {idx}"

    shutil.rmtree(temp_checkpoint_folder_path)
    shutil.rmtree(temp_history_folder_path)
    

if __name__ == '__main__':
    test_history_hook()
