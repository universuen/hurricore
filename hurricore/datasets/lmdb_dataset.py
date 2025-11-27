from __future__ import annotations

import os
import datetime
import json
import shutil
import warnings
from typing import Any
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import lmdb
from torch.utils.data import Dataset
from tqdm import tqdm


class LMDBDataset(Dataset):
    def __init__(
        self, 
        lmdb_path: str | Path = None,
        lmdb_paths: list[str | Path] = None, 
        max_num_readers: int = None,
    ) -> None:
        super().__init__()
        if (lmdb_path is None and lmdb_paths is None) or (lmdb_path is not None and lmdb_paths is not None):
            raise ValueError("Only one of `lmdb_path` or `lmdb_paths` should be provided.")
        
        self.lmdb_paths = [str(lmdb_path)] if lmdb_path is not None else [str(p) for p in lmdb_paths]
        self.max_num_readers = max_num_readers or os.cpu_count()
        self.label_statistics = dict()
        self.label_to_indices = dict()
        
        # This will be initialized lazily in each worker process
        self.lmdb_envs = None

        # We still need to compute the lengths, so we open the environments
        # temporarily in the main process for this purpose.
        self.lengths = []
        for path in self.lmdb_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"LMDB not found at {path}. Please build the LMDB first.")
            with lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False) as env:
                with env.begin(write=False) as txn:
                    self.lengths.append(txn.stat()['entries'])
            try:
                with open(Path(path) / 'meta.json', 'r') as f:
                    meta = json.load(f)
                label_statistics = meta["label_statistics"]
                for label, count in label_statistics.items():
                    self.label_statistics[label] = self.label_statistics.get(label, 0) + count
                label_to_indices = meta["label_to_indices"]
                for label, indices in label_to_indices.items():
                    real_indices = [i + sum(self.lengths) - self.lengths[-1] for i in indices]
                    self.label_to_indices[label] = self.label_to_indices.get(label, []) + real_indices
            except (FileNotFoundError, KeyError) as e:
                warnings.warn(f"Error loading meta.json from {path}: {e}")
        
        self._length = sum(self.lengths)
        self._cumulative_lengths = [sum(self.lengths[:i]) for i in range(len(self.lengths) + 1)]


    def _init_envs(self):
        """Initializes the LMDB environments for the current worker."""
        self.lmdb_envs = [
            lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=self.max_num_readers, map_size=int(1e9))
            for path in self.lmdb_paths
        ]


    def __len__(self) -> int:
        return self._length


    def __getitem__(self, index_or_indices: int) -> Any:
        # Lazy initialization: if the environments are not open, open them.
        # This will be called once per worker process.
        if self.lmdb_envs is None:
            self._init_envs()

        if isinstance(index_or_indices, slice):
            # Create a range of integer indices from the slice
            indices = range(*index_or_indices.indices(len(self)))
            # Recursively call __getitem__ for each integer index and return a list
            return [self[i] for i in indices]

        index = index_or_indices
        
        if index < 0:
            index = len(self) + index
        if index >= len(self):
            raise IndexError("Index out of range")
        
        # Determine which LMDB dataset contains the index
        lmdb_idx = next(i for i, cum_len in enumerate(self._cumulative_lengths) if index < cum_len) - 1
        dataset_offset = self._cumulative_lengths[lmdb_idx]
        local_index = index - dataset_offset
        
        # Use the worker-specific environment
        env = self.lmdb_envs[lmdb_idx]
        with env.begin(write=False) as txn:
            key = str(local_index).encode('ascii')
            value_bytes = txn.get(key)
        
        if value_bytes is None:
            # This can happen if the DB is corrupted or keys are missing
            raise KeyError(f"Key {local_index} not found in LMDB {self.lmdb_paths[lmdb_idx]}")

        item = pickle.loads(value_bytes)
        
        return item


    @classmethod
    def build(
        cls, 
        src_dataset: Dataset, 
        dst_lmdb_path: str | Path, 
        drop_none_values: bool = True,
        num_workers: int = None,
        get_label_fn: callable = None,
        flush_interval: int = 1000,
        estimated_sample_size: int = 100 * (1024 ** 2),  # 100 MB
    ) -> 'LMDBDataset':
        num_workers = num_workers or os.cpu_count()
        if get_label_fn is None:
            warnings.warn("No label function provided, meta information will be incomplete.")
        dst_lmdb_path = Path(dst_lmdb_path)
        if dst_lmdb_path.exists():
            shutil.rmtree(dst_lmdb_path)
        dst_lmdb_path.mkdir(parents=True)
        
        num_samples = len(src_dataset)
        lmdb_env_map_size = num_samples * estimated_sample_size
        
        data_queue = Queue(maxsize=num_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            writer_future = executor.submit(
                cls._writer_fn,
                lmdb_path=dst_lmdb_path,
                map_size=lmdb_env_map_size,
                flush_interval=flush_interval,
                data_queue=data_queue,
                get_label_fn=get_label_fn,
            )
            
            # Setup the reader processes
            reader_futures = (
                executor.submit(
                    cls._reader_fn, 
                    src_dataset=src_dataset,
                    index=i,
                    drop_none_values=drop_none_values,
                    data_queue=data_queue,
                )
                for i in tqdm(range(num_samples), desc="Submitting Tasks")
            )
            
            # Wait for all reader tasks to complete.
            # This loop also ensures any exceptions from readers are raised.
            for future in tqdm(as_completed(reader_futures), total=num_samples, desc="Building LMDB"):
                future.result()
            
            # Signal the writer process that all data has been sent
            data_queue.put(None)
            
            # .result() blocks until the future is complete and returns the function's result.
            # It will also re-raise any exception that occurred in the writer thread.
            label_statistics, label_to_indices = writer_future.result()
        cls._create_meta_info(
            label_statistics=label_statistics,
            label_to_indices=label_to_indices,
            dst_lmdb_path=dst_lmdb_path,
        )
        tgt_dataset = cls(dst_lmdb_path)
        return tgt_dataset

    
    @staticmethod
    def _create_meta_info(label_statistics: dict, label_to_indices: dict, dst_lmdb_path: str | Path) -> None:
        meta_info = {
            "creation_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": sum(label_statistics.values()),
            "label_statistics": dict(sorted(label_statistics.items(), key=lambda item: item[1], reverse=True)),
            "label_to_indices": dict(sorted(label_to_indices.items(), key=lambda item: len(item[1]), reverse=True)),
        }
        # Create a temporary copy of meta_info to modify before dumping
        meta_info_for_dump = meta_info.copy()
        replacements = {}

        # Check if the label_to_indices key exists and is not empty
        if "label_to_indices" in meta_info_for_dump and meta_info_for_dump["label_to_indices"]:
            placeholder_dict = {}
            for i, (label, indices) in enumerate(meta_info_for_dump["label_to_indices"].items()):
                placeholder = f"__COMPACT_LIST_PLACEHOLDER_{i}__"
                compact_list_str = f"[{', '.join(map(str, indices))}]"
                placeholder_in_json = f'"{placeholder}"'
                replacements[placeholder_in_json] = compact_list_str
                placeholder_dict[label] = placeholder
            meta_info_for_dump["label_to_indices"] = placeholder_dict

        json_string = json.dumps(
            meta_info_for_dump, 
            indent=4, 
            ensure_ascii=False
        )
        for placeholder, compact_list in replacements.items():
            json_string = json_string.replace(placeholder, compact_list)
        with open(dst_lmdb_path / "meta.json", "w", encoding="utf-8") as f:
            f.write(json_string)
        


    @classmethod
    def _writer_fn(
        cls, 
        lmdb_path: str, 
        map_size: int, 
        flush_interval: int,
        data_queue: Queue,
        get_label_fn: callable = None,
    ) -> tuple[dict, dict]:
        label_statistics_map = {}
        label_to_indices_map = {}
        idx = 0
        with lmdb.Environment(str(lmdb_path), readonly=False, map_size=map_size) as lmdb_env:
            txn = lmdb_env.begin(write=True)
            while True:
                data = data_queue.get()
                if data is None:  # Signal to stop
                    break
                
                _, item = data
                key = str(idx).encode('ascii')
                value = pickle.dumps(item)
                txn.put(key, value)
                
                if get_label_fn is None:
                    label_statistics_map["default"] = label_statistics_map.get("default", 0) + 1
                else:
                    try:
                        label = str(get_label_fn(item))
                        label_statistics_map[label] = label_statistics_map.get(label, 0) + 1
                        label_to_indices_map.setdefault(label, []).append(idx)
                    except Exception as e:
                        label_statistics_map["default"] = label_statistics_map.get("default", 0) + 1
                        print(f"Error getting label for index {idx}: {e}")

                idx += 1
                if idx % flush_interval == 0:  
                    txn.commit() 
                    txn = lmdb_env.begin(write=True)
            
            txn.commit()
            
        return label_statistics_map, label_to_indices_map


    @classmethod
    def _reader_fn(
        cls,
        src_dataset: Dataset,
        index: int,
        drop_none_values: bool,
        data_queue: Queue,
    ) -> None:
        try:
            item = src_dataset[index]
            if drop_none_values and item is None:
                return
            data_queue.put((index, item))
        except Exception as e:
            # Handle potential errors during item retrieval
            print(f"Error processing index {index}: {e}")
