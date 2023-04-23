import unittest
import time
import os
from src.dataset import LLMClassifierDataset


class TestLLMClassifierDataset(unittest.TestCase):
    """
    Test that the LLMClassifierDataset class works correctly.

    Attributes:
        db_path: (class attribute) The path to the database for the test, from the perspective of this test.
        dataset: The dataset object to be tested.
    """
    db_path: str = 'test_db.sqlite3'
    dataset: LLMClassifierDataset

    def setUp(self):
        """
        Sets up for each test, by creating the dataset.
        """
        self.dataset = LLMClassifierDataset(self.db_path)

    def tearDown(self):
        """
        Tears down after each set, deleting the dataset and the test database.
        """
        del self.dataset
        self.dataset = None
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


if __name__ == '__main__':
    unittest.main()
