import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricore.trainers import Trainer
from hurricore.hooks import SyncBatchNormHook


class _TestTrainer(Trainer):
    def __init__(self):
        super().__init__(
            models=[
                nn.Sequential(
                    nn.BatchNorm1d(1),
                    nn.BatchNorm2d(1),
                    nn.BatchNorm3d(1),
                    nn.ModuleList([nn.BatchNorm1d(1), nn.BatchNorm2d(1), nn.BatchNorm3d(1)]),
                    nn.ModuleDict({'bn1d': nn.BatchNorm1d(1), 'bn2d': nn.BatchNorm2d(1), 'bn3d': nn.BatchNorm3d(1)}),
                    nn.Sequential(nn.BatchNorm1d(1), nn.BatchNorm2d(1), nn.BatchNorm3d(1)),
                )
            ],
            optimizers=[AdamW(nn.Linear(1, 1).parameters(), lr=1e-3)],
            data_loaders=[DataLoader(range(1), batch_size=1, shuffle=True)],
            accelerator=Accelerator(),
            num_epochs=1,
        )
        self.hooks = [SyncBatchNormHook(self)]
        
    
    def compute_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, requires_grad=True)


def test_sync_batchnorm_hook():
    trainer = _TestTrainer()
    trainer.run()
    def check_sync_batchnorm(model):
        for module in model.modules():
            if isinstance(module, nn.SyncBatchNorm):
                return True
        return False
    assert all(check_sync_batchnorm(model) for model in trainer.models), "Not all BatchNorm modules are converted to SyncBatchNorm."
