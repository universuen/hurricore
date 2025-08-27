from __future__ import annotations

import os
import json
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
import pickle

from torch.utils.data import Dataset
from tqdm import tqdm

from hurricore.utils import chunkify


class PackedDataset(Dataset):
    def __init__(
        self, 
        dataset: Dataset, 
        pack_dir_path: str | Path,
        chunk_size: int = 100
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pack_dir_path = Path(pack_dir_path)
        self.chunk_size = chunk_size
        self.packs = sorted(self.pack_dir_path.glob("*.pkl"), key=lambda x: x.name)
        self.index_map_path = self.pack_dir_path / "index_map.json"
        
        if self.index_map_path.exists():
            with open(self.index_map_path, 'r') as f:
                self.index_map: list[tuple[int, int]] = json.load(f)
        else:
            self._build_packs()
            self._build_index_map()


    def __len__(self) -> int:
        return len(self.index_map)


    def __getitem__(self, index: int) -> tuple:
        pack_index, record_index_in_pack = self.index_map[index]
        pack_path = self.packs[pack_index]
        with open(pack_path, 'rb') as f:
            records: list[tuple] = pickle.load(f)
        return records[record_index_in_pack]


    def _build_index_map(self) -> None:
        print("Building index map...")

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            imaps = list(tqdm(executor.map(self._process_pack_for_index, enumerate(self.packs)), total=len(self.packs)))

        index_map = []
        for imap in imaps:
            index_map.extend(imap)

        with open(self.index_map_path, 'w') as f:
            json.dump(index_map, f)

        self.index_map = index_map


    @staticmethod
    def _process_pack_for_index(pack_tuple: tuple[int, Path]) -> list[tuple[int, int]]:
        pack_index, pack_path = pack_tuple
        with open(pack_path, 'rb') as f:
            records: list[tuple] = pickle.load(f)
        num_records = len(records)
        return [(pack_index, i) for i in range(num_records)]


    def _build_packs(self) -> None:
        indices = list(range(len(self.dataset)))
        splitted_chunks = chunkify(indices, self.chunk_size)
        num_packs = len(splitted_chunks)
        pack_file_paths = [self.pack_dir_path / f'pack_{i:08d}.pkl' for i in range(num_packs)]

        print('Building packs...')

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm(executor.map(self._process_chunk, zip(splitted_chunks, pack_file_paths)), total=num_packs))
        
        self.packs = sorted(self.pack_dir_path.glob("*.pkl"), key=lambda x: x.name)


    def _process_chunk(self, indices_pack_pair: tuple[list[int], Path]) -> None:
        indices, pack_file_path = indices_pack_pair
        check_file_path = pack_file_path.with_suffix('.check')
        if check_file_path.exists():
            print(f"Pack file {pack_file_path} already exists. Skipping...")
            return
        records: list[tuple] = []
        for idx in indices:
            record = self.dataset[idx]
            records.append(record)
        with open(pack_file_path, 'wb') as f:
            pickle.dump(records, f, protocol=5)
        with open(check_file_path, 'w') as f:
            f.write('done')
        
