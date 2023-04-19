import unittest
import os
from src.dataset import generate_datasets_for_llm, InferenceLLM, LLMClassifierDataset


class TestGenerateDatasets(unittest.TestCase):
    """
    Tests that generating datasets works correctly

    Attributes:
        test_db_path: (class attribute) The path to the database for the test set.
        train_db_path: (class attribute) The path to the database for the train set.
        ms_marco_file: (class attribute) The path to the mock ms marco set.
    """
    test_db_path: str = 'test_short_prompts.sqlite3'
    train_db_path: str = 'train_short_prompts.sqlite3'
    ms_marco_file: str = 'mock_data_ms_marco.json'

    def tearDown(self):
        """
        Deletes the temporary databases.
        """
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists(self.train_db_path):
            os.remove(self.train_db_path)

    def test_generate_datasets_for_bloom_1_1B(self):
        """
        Test that generating the datasets for bloom_1_1B works correctly.
        """
        llm = InferenceLLM('bigscience/bloom-1b1')
        generate_datasets_for_llm(llm, './', self.ms_marco_file, self.ms_marco_file)
        test_dataset = LLMClassifierDataset(self.test_db_path)
        self.assertEqual(20, len(test_dataset))
        train_dataset = LLMClassifierDataset(self.train_db_path)
        self.assertEqual(20, len(train_dataset))

    def test_generate_datasets_for_opt_1_3B(self):
        """
        Test that generating the datasets for bloom_1_1B works correctly.
        """
        llm = InferenceLLM('opt-1.3b')
        generate_datasets_for_llm(llm, './', self.ms_marco_file, self.ms_marco_file)
        test_dataset = LLMClassifierDataset(self.test_db_path)
        self.assertEqual(20, len(test_dataset))
        train_dataset = LLMClassifierDataset(self.train_db_path)
        self.assertEqual(20, len(train_dataset))
