import unittest
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

    def test_generate_llm_prompts(self):
        """
        Tests that generate_llm_prompts correctly converts DB rows into prompts, answer lengths, and sorted indices.
        """
        pass

    def test_prompt(self):
        """
        Tests that prompt correctly takes passages and a query and converts it to a LLM prompt.
        """
        pass

    def test_sort_array_value(self):
        """
        Tests that sort_array_value correctly sorts the given list and returns the sorted indexes.
        """
        pass

    def test_sort_array_indices(self):
        """
        Tests that sort_array_indices correctly sorts an array according to the given indices.
        """
        pass

    def test_unsort_array(self):
        """
        Tests that unsort_array correctly returns an array to the order it was in before being sorted.
        """
        pass

    def test_llm_answers_to_db(self):
        """
        Tests that the LLM is run, and that the answers are progressively added to the database as a side effect.
        """
        pass

if __name__ == '__main__':
    unittest.main()
