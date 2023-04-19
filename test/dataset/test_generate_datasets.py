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
    test_db_path: str = '../../test/dataset/test_short_prompts.sqlite3'
    train_db_path: str = '../../test/dataset/train_short_prompts.sqlite3'
    ms_marco_file: str = '../../test/dataset/mock_data_ms_marco.json'

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
        generate_datasets_for_llm(llm=llm, db_folder='../../test/dataset', batch_size=8,
                                  ms_marco_test=self.ms_marco_file, ms_marco_train=self.ms_marco_file)
        test_dataset = LLMClassifierDataset(self.test_db_path)
        self.assertEqual(22, len(test_dataset))
        train_dataset = LLMClassifierDataset(self.train_db_path)
        self.assertEqual(22, len(train_dataset))
        # Deleting the datasets when we are done, so the db connection closes
        del test_dataset
        del train_dataset
