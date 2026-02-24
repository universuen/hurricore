from typing import Any

from torch.utils.data import Dataset
import random


class ShuffledDataset(Dataset):
    def __init__(self, dataset: Dataset, seed: int = 42):
        super().__init__()
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        random.seed(seed)
        random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        return self.dataset[self.indices[idx]]


if __name__ == '__main__':
    class MyDS(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]
    
    ds = MyDS(list(range(10)))
    shuffled_ds = ShuffledDataset(ds)
    for i in ds:
        print(i, shuffled_ds[i])