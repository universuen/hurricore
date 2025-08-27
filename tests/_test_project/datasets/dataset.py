from __future__ import annotations

import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(100, 32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], torch.randint(0, 10, (1,)).item()
