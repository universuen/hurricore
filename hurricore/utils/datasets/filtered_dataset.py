from typing import Callable, Any
from torch.utils.data import Dataset

class FilteredDataset(Dataset):
    def __init__(
            self, 
            dataset: Dataset, 
            filter_fn: Callable[[Any], bool],
            default_value: Any = None,
        ):
        """
        Initializes the FilteredDataset with a dataset and a filtering function.

        Args:
            dataset (Dataset): The original dataset to be filtered.
            filter_fn (Callable): A function that takes a dataset item as input and returns
                                  a boolean indicating whether the item should be included.
            default_value (Any, optional): The value to return if the item does not meet the filter criteria. Defaults to None.
        """
        self.dataset = dataset
        self.filter_fn = filter_fn
        self.default_value = default_value


    def __getitem__(self, index):
        item = self.dataset[index]
        if self.filter_fn(item):
            return item
        else:
            return self.default_value
        

    def __len__(self):
        return len(self.dataset)
