import random
from typing import Any

from torch.utils.data import Dataset


class ContinualLearningDataset(Dataset):
    def __init__(
        self,
        memory_dataset: Dataset,
        update_dataset: Dataset,
        memory_propotion: float = 0.5,
    ) -> None:
        super().__init__()
        self.memory_dataset = memory_dataset
        self.update_dataset = update_dataset
        self.memory_propotion = memory_propotion
        self.num_update_samples = len(update_dataset)
        self.num_memory_samples = int((memory_propotion / (1 - memory_propotion)) * len(update_dataset))
    

    def __len__(self) -> int:
        return self.num_update_samples + self.num_memory_samples
    

    def __getitem__(self, index: int) -> Any:
        return self.update_dataset[index] if index < self.num_update_samples else random.choice(self.memory_dataset)
