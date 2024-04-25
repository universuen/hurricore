import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pass

