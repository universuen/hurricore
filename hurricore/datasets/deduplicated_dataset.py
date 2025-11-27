from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from hurricore.utils.functions import hash_anything


class DeduplicatedDataset(Dataset):
    def __init__(self, dataset: Dataset, get_content_func: Callable = None):
        self.dataset = dataset
        self.get_content_func = get_content_func
        self.hash_original_index_map = self._build_hash_original_index_map()
        self.new_index_hash_map = {
            index: hash_str for index, hash_str in enumerate(self.hash_original_index_map.keys())
        }
        
        
    def _build_hash_original_index_map(self) -> dict:
        hash_map = {}
        for i, item in tqdm(enumerate(self.dataset), desc="Deduplicating", total=len(self.dataset)):
            content = item if self.get_content_func is None else self.get_content_func(item)
            hash_str = hash_anything(content)
            hash_map[hash_str] = i
        return hash_map
    

    def __len__(self):
        return len(self.hash_original_index_map)
    

    def __getitem__(self, index):
        if index < 0:
            index = len(self) + index
        if index >= len(self):
            raise IndexError("Index out of range")
        hash_str = self.new_index_hash_map[index]
        original_index = self.hash_original_index_map[hash_str]
        return self.dataset[original_index]
