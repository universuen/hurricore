import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator

from hurricore.trainers import HurricoreTrainer
from hurricore.hooks import LRSchedulerHook


class _TestTrainer(HurricoreTrainer):
    def __init__(
        self,
        num_processes = 1,
        # gradient_accumulation_steps = 10,
        batch_size = 8,
        num_epochs = 2,
    ):
        model = nn.Linear(1, 1)
        optimizer = AdamW(model.parameters(), lr=1)
        data_loader = DataLoader(range(100), batch_size=batch_size, shuffle=True)
        super().__init__(
            models=[model],
            optimizers=[optimizer],
            data_loaders=[data_loader],
            accelerator=Accelerator(),
            num_epochs=num_epochs,
            hooks=[
                LRSchedulerHook(
                    trainer=self,
                    lr_schedulers=[
                        CosineAnnealingLR(
                            optimizer=optimizer,
                            T_max=num_epochs * len(data_loader) * num_processes,
                            eta_min=0.0,
                        )
                    ],
                    modes=['per_step']
                )
            ],
        )
        self.iterated_results = []
    
    
    def step(self, batch) -> torch.Tensor:
        self.optimizers[0].step()
        return torch.tensor(0.0)
    

def test_lr_scheduler_hook():
    trainer = _TestTrainer()
    trainer.run()
    lr_scheduler = trainer.hooks["LRSchedulerHook"].lr_schedulers[0]
    current_lr = lr_scheduler.get_last_lr()[0]
    assert current_lr == 0.0, \
        (
            f"Learning rate is not scheduled as expected.\n"
            f"\tExpected 0.0, got {current_lr}."
        )


if __name__ == '__main__':
    test_lr_scheduler_hook()
