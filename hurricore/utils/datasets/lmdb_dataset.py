from __future__ import annotations

import os
import datetime
import json
import shutil
from typing import Any
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from queue import Queue

import lmdb
from torch.utils.data import Dataset
from tqdm import tqdm


class LMDBDataset(Dataset):
    def __init__(
        self, 
        lmdb_path: str | Path = None,
        lmdb_paths: list[str | Path] = None, 
        max_num_readers: int = os.cpu_count(),
    ) -> None:
        super().__init__()
        if (lmdb_path is None and lmdb_paths is None) or (lmdb_path is not None and lmdb_paths is not None):
            raise ValueError("Only one of `lmdb_path` or `lmdb_paths` should be provided.")
        
        self.lmdb_paths = [str(lmdb_path)] if lmdb_path is not None else [str(p) for p in lmdb_paths]
        self.max_num_readers = max_num_readers
        
        # This will be initialized lazily in each worker process
        self.lmdb_envs = None

        # We still need to compute the lengths, so we open the environments
        # temporarily in the main process for this purpose.
        self.lengths = []
        for path in self.lmdb_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"LMDB not found at {path}. Please build the LMDB first.")
            env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                self.lengths.append(txn.stat()['entries'])
            env.close() # Close the handle immediately
        
        self._length = sum(self.lengths)
        self.cumulative_lengths = [sum(self.lengths[:i]) for i in range(len(self.lengths) + 1)]


    def _init_envs(self):
        """Initializes the LMDB environments for the current worker."""
        self.lmdb_envs = [
            lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=self.max_num_readers)
            for path in self.lmdb_paths
        ]


    def __len__(self) -> int:
        return self._length


    def __getitem__(self, index: int) -> Any:
        # Lazy initialization: if the environments are not open, open them.
        # This will be called once per worker process.
        if self.lmdb_envs is None:
            self._init_envs()

        if index < 0:
            index = len(self) + index
        if index >= len(self):
            raise IndexError("Index out of range")
        
        # Determine which LMDB dataset contains the index
        lmdb_idx = next(i for i, cum_len in enumerate(self.cumulative_lengths) if index < cum_len) - 1
        dataset_offset = self.cumulative_lengths[lmdb_idx]
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
        tgt_lmdb_path: str | Path, 
        drop_none_values: bool = True,
        num_workers: int = os.cpu_count(),
        get_label: callable = None,
        flush_interval: int = 1000,
        estimated_sample_size: int = 10 * (1024 ** 2),  # 10 MB
    ) -> 'LMDBDataset':
        tgt_lmdb_path = Path(tgt_lmdb_path)
        if tgt_lmdb_path.exists():
            shutil.rmtree(tgt_lmdb_path)
        tgt_lmdb_path.mkdir(parents=True)
        
        num_samples = len(src_dataset)
        lmdb_env_map_size = num_samples * estimated_sample_size
        
        # Create queue with appropriate size
        data_queue = Queue(maxsize=num_workers)
        
        # Use ThreadPoolExecutor context manager
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Setup the writer process
            writer_thread = Thread(
                target=cls._writer_fn,
                kwargs={
                    "lmdb_path": tgt_lmdb_path,
                    "map_size": lmdb_env_map_size,
                    "flush_interval": flush_interval,
                    "data_queue": data_queue,
                }
            )
            writer_thread.start()
            
            # Setup the reader processes
            futures = (
                executor.submit(
                    cls._reader_fn, 
                    src_dataset=src_dataset,
                    index=i,
                    drop_none_values=drop_none_values,
                    data_queue=data_queue,
                    get_label=get_label,
                )
                for i in tqdm(range(num_samples), desc="Submitting Tasks")
            )
            
            label_statistics = {}
            for future in tqdm(as_completed(futures), total=num_samples, desc="Building LMDB"):
                label = future.result()  # Raise any exceptions
                if label is not None:
                    label_statistics[label] = label_statistics.get(label, 0) + 1
            
            # Signal the writer process that all data has been sent
            data_queue.put(None)
            writer_thread.join()

        meta_info = {
            "creation_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": sum(label_statistics.values()),
            "label_statistics": dict(sorted(label_statistics.items(), key=lambda item: item[1], reverse=True)),
        }
        with open(tgt_lmdb_path / "meta.json", "w") as f:
            json.dump(meta_info, f, indent=4, ensure_ascii=False)

        # Return the built LMDBDataset
        return cls(tgt_lmdb_path)


    @classmethod
    def _writer_fn(
        cls, 
        lmdb_path: str, 
        map_size: int, 
        flush_interval: int,
        data_queue: Queue
    ) -> None:
        idx = 0
        with lmdb.Environment(str(lmdb_path), readonly=False, map_size=map_size) as lmdb_env:
            txn = lmdb_env.begin(write=True)
            while True:
                data = data_queue.get()
                if data is None:  # Signal to stop
                    break
                
                _, item = data  # Ignore placeholder
                key = str(idx).encode('ascii')
                value = pickle.dumps(item)
                txn.put(key, value)
                
                idx += 1
                if idx % flush_interval == 0:  
                    txn.commit() 
                    txn = lmdb_env.begin(write=True)
            
            txn.commit()


    @classmethod
    def _reader_fn(
        cls,
        src_dataset: Dataset,
        index: int,
        drop_none_values: bool,
        data_queue: Queue,
        get_label: callable,
    ) -> Any:
        try:
            item = src_dataset[index]
            if drop_none_values and item is None:
                return None
            data_queue.put((index, item))
            return None if get_label is None else get_label(item)
        except Exception as e:
            # Handle potential errors during item retrieval
            print(f"Error processing index {index}: {e}")
