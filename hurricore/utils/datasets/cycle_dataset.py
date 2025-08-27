from typing import Any

from torch.utils.data import Dataset

class CycleDataset(Dataset):
    def __init__(self, dataset: Dataset, length: int) -> None:
        """
        Initialize the CycleDataset with an iterable dataset and a desired length.

        :param dataset: An iterable dataset to cycle through
        :param length: Total number of items to provide
        """
        self.dataset = dataset
        if not self.dataset:
            raise ValueError("Dataset must not be empty.")
        self.length = length

    def __getitem__(self, index: int) -> Any:
        """
        Get the item at the given index, cycling through the dataset if needed.

        :param index: Index of the item to retrieve
        :return: The item at the given index
        """
        if index < 0 or index >= self.length:
            raise IndexError("Index out of range.")
        return self.dataset[index % len(self.dataset)]

    def __len__(self) -> int:
        """
        Return the total length of the cycled dataset.

        :return: Length of the dataset
        """
        return self.length
