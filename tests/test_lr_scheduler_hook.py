import _path_setup  # noqa: F401

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricane.trainers import Trainer
from hurricane.hooks import LRSchedulerHook


class _TestTrainer(Trainer):
    def __init__(
        self,
        num_processes = 1,
        # gradient_accumulation_interval = 10,
        batch_size = 8,
        epochs = 2,
    ):
        model = nn.Linear(1, 1)
        optimizer = AdamW(model.parameters(), lr=1)
        super().__init__(
            models=[model],
            optimizers=[optimizer],
            data_loaders=[
                DataLoader(
                    range(100), 
                    batch_size=batch_size, 
                    shuffle=True
                ),
            ],
            accelerator=Accelerator(),
            epochs=epochs,
        )
        num_batches = len(self.data_loaders[0])
        self.hooks = [
            LRSchedulerHook(
                trainer=self,
                lr_schedulers=[
                    CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=epochs * num_batches * num_processes,
                        eta_min=0.0,
                    )
                ],
                mode='per_step'
            )
        ]
        self.iterated_results = []
    
    
    def training_step(self) -> torch.Tensor:
        self.optimizers[0].step()
        return torch.tensor(0.0)
    

def test_lr_scheduler_hook():
    trainer = _TestTrainer()
    trainer.run()
    lr_scheduler = trainer.get_hook(LRSchedulerHook).originals.lr_schedulers[0]
    current_lr = lr_scheduler.get_last_lr()[0]
    assert current_lr == 0.0, \
        (
            f"Learning rate is not scheduled as expected.\n"
            f"\tExpected 0.0, got {current_lr}."
        )
