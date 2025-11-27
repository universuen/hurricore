from __future__ import annotations

import random
from typing import Any, Callable


from torch.utils.data import Dataset

from hurricore.utils import build_label_indices_map


class BalancedDataset:
    def __init__(
        self,
        dataset: Dataset,
        get_label: Callable[[Any], Any] = lambda x: x[1],
        num_samples_per_label: int = None,
        batch_size: int = 32,
    ):
        self.dataset = dataset
        self.get_label = get_label
        self.num_samples_per_label = num_samples_per_label

        self.label_index = build_label_indices_map(
            dataset=self.dataset,
            get_label=self.get_label,
            batch_size=batch_size,
        )
        self._balance_samples_per_label()


    def _balance_samples_per_label(self):
        if self.num_samples_per_label is None:
            self.num_samples_per_label = max(len(v) for v in self.label_index.values())
        for label, indices in self.label_index.items():
            if len(indices) < self.num_samples_per_label:
                indices += random.choices(indices, k=self.num_samples_per_label - len(indices))
            else:
                indices = indices[:self.num_samples_per_label]
            self.label_index[label] = indices


    def __len__(self) -> int:
        return len(self.label_index) * self.num_samples_per_label


    def __getitem__(self, index: int) -> Any:
        label_index = index // self.num_samples_per_label
        sample_index = index % self.num_samples_per_label
        label = list(self.label_index.keys())[label_index]
        return self.dataset[self.label_index[label][sample_index]]


if __name__ == '__main__':
    test_dataset = [
        (0, 0),
        (1, 1),
        (2, 1),
        (1, 1),
        (1, 2),
    ]
    test_balanced_lmdb = BalancedDataset(test_dataset)
    print(len(test_balanced_lmdb))
    for i in range(len(test_balanced_lmdb)):
        print(test_balanced_lmdb[i])
