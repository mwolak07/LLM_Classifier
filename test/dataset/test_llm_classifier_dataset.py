import unittest
import os
from src.dataset import LLMClassifierDataset, LLMClassifierRow


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
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_generate_llm_prompts(self):
        """
        Tests that generate_llm_prompts correctly converts DB rows into prompts, answer lengths, and sorted indices.
        """
        # Creating fake DB rows.
        rows = []
        query1 = 'query1?'
        passages1 = ['Test passage 1', 'Test passage 2']
        human_answer1 = 'Human answer1! This is a long one.'
        rows.append(LLMClassifierRow(query1, passages1, None, human_answer1, None, True))
        query2 = 'query2'
        passages2 = ['Test passage 3', 'Test passage 4', 'Test passage 5']
        human_answer2 = 'Answer2.'
        rows.append(LLMClassifierRow(query2, passages2, None, human_answer2, None, True))
        query3 = 'query3?'
        passages3 = ['Test passage 6']
        human_answer3 = 'Human answer3, medium'
        rows.append(LLMClassifierRow(query3, passages3, None, human_answer3, None, True))
        # Generating the expected answer.
        prompts = [LLMClassifierDataset.prompt(passages2, query2),
                   LLMClassifierDataset.prompt(passages3, query3),
                   LLMClassifierDataset.prompt(passages1, query1)]
        answer_lengths = [len(human_answer2), len(human_answer3), len(human_answer1)]
        sorted_indices = [1, 2, 0]
        # Checking the method.
        self.assertEqual((prompts, answer_lengths, sorted_indices), self.dataset.generate_llm_prompts(rows))

    def test_prompt(self):
        """
        Tests that prompt correctly takes passages and a query and converts it to a LLM prompt.
        """
        # Normal case, with ?.
        passages = ['This is a sample passage.', 'This is another sample passage.']
        query = 'What is a sample passage?'
        expected_prompt = 'Using only the following context:\nThis is a sample passage.\n\nThis is another sample ' \
                          'passage.\n\nThe short answer, in complete sentences, to the question: "What is a sample ' \
                          'passage?", is:\n'

        prompt = self.dataset.prompt(passages, query)
        self.assertEqual(expected_prompt, prompt)

        # Normal case, adds ?.
        passages = ['This is a sample passage.', 'This is another sample passage.']
        query = 'What is a sample passage'
        expected_prompt = 'Using only the following context:\nThis is a sample passage.\n\nThis is another sample ' \
                          'passage.\n\nThe short answer, in complete sentences, to the question: "What is a sample ' \
                          'passage?", is:\n'
        prompt = self.dataset.prompt(passages, query)
        self.assertEqual(expected_prompt, prompt)

        # Dealing with empty query correctly
        passages = ['This is a sample passage.', 'This is another sample passage.']
        query = ''
        expected_prompt = 'Using only the following context:\nThis is a sample passage.\n\nThis is another sample ' \
                          'passage.\n\nThe short answer, in complete sentences, to the question: "?", is:\n'
        prompt = self.dataset.prompt(passages, query)
        self.assertEqual(expected_prompt, prompt)

        # Dealing with empty passages correctly
        passages = []
        query = 'What is a sample passage?'
        expected_prompt = 'Using only the following context:\nThe short answer, in complete sentences, to the ' \
                          'question: "What is a sample passage?", is:\n'
        prompt = self.dataset.prompt(passages, query)
        self.assertEqual(expected_prompt, prompt)

    def test_sort_array_value(self):
        """
        Tests that sort_array_value correctly sorts the given list and returns the sorted indexes.
        """
        # Reverse order case.
        test = [6, 5, 4, 3, 2, 1]
        test_indices = [5, 4, 3, 2, 1, 0]
        test_sorted = [1, 2, 3, 4, 5, 6]
        self.assertEqual((test_sorted, test_indices), LLMClassifierDataset.sort_array_value(test))
        # Random order case.
        test = ['l', 'c', 'a', 'z', 'k', 'm']
        test_indices = [2, 1, 4, 0, 5, 3]
        test_sorted = ['a', 'c', 'k', 'l', 'm', 'z']
        self.assertEqual((test_sorted, test_indices), LLMClassifierDataset.sort_array_value(test))
        # Sorted order case.
        test = ['a', 'd', 'e', 'g', 'm', 'q']
        test_indices = [0, 1, 2, 3, 4, 5]
        test_sorted = ['a', 'd', 'e', 'g', 'm', 'q']
        self.assertEqual((test_sorted, test_indices), LLMClassifierDataset.sort_array_value(test))

    def test_sort_array_indices(self):
        """
        Tests that sort_array_indices correctly sorts an array according to the given indices.
        """
        # Reverse order case.
        test = [6, 5, 4, 3, 2, 1]
        test_indices = [5, 4, 3, 2, 1, 0]
        test_sorted = [1, 2, 3, 4, 5, 6]
        self.assertEqual(test_sorted, LLMClassifierDataset.sort_array_indices(test, test_indices))
        # Random order case.
        test = ['a', 'b', 'c', 'd', 'e', 'f']
        test_indices = [2, 1, 4, 0, 5, 3]
        test_sorted = ['c', 'b', 'e', 'a', 'f', 'd']
        self.assertEqual(test_sorted, LLMClassifierDataset.sort_array_indices(test, test_indices))
        # Same order case.
        test = [1, 2, 3, 4, 5, 6]
        test_indices = [0, 1, 2, 3, 4, 5]
        test_sorted = [1, 2, 3, 4, 5, 6]
        self.assertEqual(test_sorted, LLMClassifierDataset.sort_array_indices(test, test_indices))

    def test_unsort_array(self):
        """
        Tests that unsort_array correctly returns an array to the order it was in before being sorted.
        """
        # Reverse order case.
        test_sorted = [1, 2, 3, 4, 5, 6]
        test_indices = [5, 4, 3, 2, 1, 0]
        test = [6, 5, 4, 3, 2, 1]
        self.assertEqual(test, LLMClassifierDataset.unsort_array(test_sorted, test_indices))
        # Random order case.
        test_sorted = ['c', 'b', 'e', 'a', 'f', 'd']
        test_indices = [2, 1, 4, 0, 5, 3]
        test = ['a', 'b', 'c', 'd', 'e', 'f']
        self.assertEqual(test, LLMClassifierDataset.unsort_array(test_sorted, test_indices))
        # Same order case.
        test_sorted = [1, 2, 3, 4, 5, 6]
        test_indices = [0, 1, 2, 3, 4, 5]
        test = [1, 2, 3, 4, 5, 6]
        self.assertEqual(test, LLMClassifierDataset.unsort_array(test_sorted, test_indices))


if __name__ == '__main__':
    unittest.main()
