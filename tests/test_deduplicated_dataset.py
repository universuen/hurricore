import unittest
from torch.utils.data import Dataset

from hurricore.datasets.deduplicated_dataset import DeduplicatedDataset, hash_anything

# Mock dataset for testing
class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TestDeduplicatedDataset(unittest.TestCase):
    def setUp(self):
        # Setup mock data with duplicate items
        self.mock_data = [1, 2, 3, 1, 2, 3, 4]
        self.dataset = MockDataset(self.mock_data)
        self.dedup_dataset = DeduplicatedDataset(self.dataset)

    def test_initialization(self):
        # After initialization, only unique items should be present
        expected_unique_data = set(self.mock_data)
        dedup_data = [self.dedup_dataset[i] for i in range(len(self.dedup_dataset))]
        self.assertEqual(set(dedup_data), expected_unique_data)

    def test_length(self):
        # Length of deduplicated dataset should match the unique items
        unique_data_length = len(set(self.mock_data))
        self.assertEqual(len(self.dedup_dataset), unique_data_length)

    def test_getitem(self):
        # Ensure each item in deduplicated dataset is retrieved correctly
        unique_items = list(set(self.mock_data))
        retrieved_items = [self.dedup_dataset[i] for i in range(len(self.dedup_dataset))]
        self.assertCountEqual(retrieved_items, unique_items)

    def test_hash_consistency(self):
        # Test that hashing is consistent for identical items
        item = self.dataset[0]  # Arbitrarily select the first item
        hash1 = hash_anything(item)
        hash2 = hash_anything(item)
        self.assertEqual(hash1, hash2, "Hashes should be consistent for identical items")

    def test_deduplication(self):
        # Test that duplicates are removed
        dedup_data = [self.dedup_dataset[i] for i in range(len(self.dedup_dataset))]
        unique_data = list(set(self.mock_data))
        self.assertCountEqual(dedup_data, unique_data)

if __name__ == "__main__":
    unittest.main()
