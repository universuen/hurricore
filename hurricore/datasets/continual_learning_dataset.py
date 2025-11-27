import random
from typing import Any

from torch.utils.data import Dataset


class ContinualLearningDataset(Dataset):
    def __init__(
        self,
        replay_dataset: Dataset,
        update_dataset: Dataset,
        replay_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.replay_dataset = replay_dataset
        self.update_dataset = update_dataset
        self.replay_ratio = replay_ratio
        self.num_update_samples = len(update_dataset)
        self.num_memory_samples = int((replay_ratio / (1 - replay_ratio)) * len(update_dataset))
    

    def __len__(self) -> int:
        return self.num_update_samples + self.num_memory_samples
    

    def __getitem__(self, index: int) -> Any:
        return self.update_dataset[index] if index < self.num_update_samples else random.choice(self.replay_dataset)
