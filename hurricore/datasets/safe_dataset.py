import random
from torch.utils.data import Dataset


class SafeDataset(Dataset):
    
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    

    def __len__(self) -> int:
        return len(self.dataset)
    

    def __getitem__(self, index: int) -> dict:
        while True:
            try:
                return self.dataset[index]
            except Exception as e:
                print(f"Error getting item at index {index}: {e}")
                index = random.randint(0, len(self.dataset) - 1)

