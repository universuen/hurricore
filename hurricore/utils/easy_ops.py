from __future__ import annotations

import torch


def get_list_mean(list_: list[int | float]) -> float:
    return sum(list_) / (len(list_) + 1e-6)


def find_start_and_end_index(long_t: torch.Tensor, short_t: torch.Tensor) -> int:
        for i in range(len(long_t) - len(short_t) + 1):
            if torch.all(long_t[i:i + len(short_t)] == short_t):
                return i, i + len(short_t)
        return -1, -1
