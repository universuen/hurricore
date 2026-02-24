from pathlib import Path
import shutil

from hurricore.datasets import LMDBDataset


def test_lmdb_dataset():
    src_dataset_a = [1, 2, 3]
    src_dataset_b = [4, 5, 6, 7]
    temp_lmdb_path_a = Path(__file__).parent / "temp_lmdb_a"
    temp_lmdb_path_b = Path(__file__).parent / "temp_lmdb_b"
    if temp_lmdb_path_a.exists():
        shutil.rmtree(temp_lmdb_path_a)
    if temp_lmdb_path_b.exists():
        shutil.rmtree(temp_lmdb_path_b)
    lmdb_a = LMDBDataset.build(src_dataset_a, temp_lmdb_path_a)
    lmdb_b = LMDBDataset.build(src_dataset_b, temp_lmdb_path_b)
    assert lmdb_a[0] == 1
    assert lmdb_b[0] == 4
    assert len(lmdb_a) == 3
    assert len(lmdb_b) == 4
    combined_lmdb = LMDBDataset(lmdb_paths=[temp_lmdb_path_a, temp_lmdb_path_b])
    assert len(combined_lmdb) == 7
    for i, item in enumerate(combined_lmdb):
        assert item == i + 1
    shutil.rmtree(temp_lmdb_path_a)
    shutil.rmtree(temp_lmdb_path_b)


if __name__ == "__main__":
    test_lmdb_dataset()
