import unittest
import os
from src.dataset import generate_datasets_for_llm, InferenceLLM, LLMClassifierDataset


class TestGenerateDatasets(unittest.TestCase):
    """
    Tests that generating datasets works correctly

    Attributes:
        test_db_path: (class attribute) The path to the database for the test set, from the perspective of
                      generate_datasets_for_llm.
        train_db_path: (class attribute) The path to the database for the train set, from the perspective of
                       generate_datasets_for_llm.
        ms_marco_file: (class attribute) The path to the mock ms marco set, from the perspective of
                       generate_datasets_for_llm.
        test_dataset: The test dataset object to be tested.
        train_dataset: The train dataset object to be tested.
    """
    test_db_path: str = '../../test/dataset/test_short_prompts.sqlite3'
    train_db_path: str = '../../test/dataset/train_short_prompts.sqlite3'
    ms_marco_file: str = '../../test/dataset/mock_data_ms_marco.json'
    test_dataset: LLMClassifierDataset
    train_dataset: LLMClassifierDataset

    def setUp(self):
        """
        Creates the datasets for the tests.
        """
        self.test_dataset = LLMClassifierDataset(self.test_db_path)
        self.train_dataset = LLMClassifierDataset(self.train_db_path)

    def tearDown(self):
        """
        Deletes the datasets and temporary database files.
        """
        del self.test_dataset
        del self.train_dataset
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
        answers = self.test_dataset._db.llm_answers()
        self.assertFalse(any(['The short answer, in complete sentences, to the question:'
                              in answer for answer in answers]))
        self.assertTrue(max([len(answer) for answer in answers]) > 1)
        self.assertEqual(20, len(self.test_dataset))
        answers = self.train_dataset._db.llm_answers()
        self.assertFalse(any(['The short answer, in complete sentences, to the question:'
                              in answer for answer in answers]))
        self.assertTrue(max([len(answer) for answer in answers]) > 1)
        self.assertEqual(20, len(self.train_dataset))


if __name__ == '__main__':
    unittest.main()
