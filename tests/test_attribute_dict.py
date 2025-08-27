import unittest
from hurricore.utils.attribute_dict import AttributeDict

class TestAttributeDict(unittest.TestCase):
    def setUp(self):
        self.attr_dict = AttributeDict()

    def test_set_and_get_attribute(self):
        self.attr_dict.test_key = "test_value"
        self.assertEqual(self.attr_dict.test_key, "test_value")
        self.assertEqual(self.attr_dict["test_key"], "test_value")

    def test_set_and_get_item(self):
        self.attr_dict["test_key"] = "test_value"
        self.assertEqual(self.attr_dict.test_key, "test_value")
        self.assertEqual(self.attr_dict["test_key"], "test_value")

    def test_no_attribute_error(self):
        _ = self.attr_dict.non_existent_key

    def test_state_dict(self):
        self.attr_dict.key1 = "value1"
        self.attr_dict.key2 = "value2"
        state = self.attr_dict.state_dict()
        self.assertIsInstance(state, dict)
        self.assertEqual(state, {"key1": "value1", "key2": "value2"})

    def test_load_state_dict(self):
        state = {"key1": "value1", "key2": "value2"}
        self.attr_dict.load_state_dict(state)
        self.assertEqual(self.attr_dict.key1, "value1")
        self.assertEqual(self.attr_dict.key2, "value2")

if __name__ == '__main__':
    unittest.main()
