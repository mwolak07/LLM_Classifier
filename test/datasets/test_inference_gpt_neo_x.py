import unittest
import json
import os
from src.datasets import InferenceGPTNeoX
from test.datasets import write_questions


class TestInferenceGPTNeoX(unittest.TestCase):
    """
    Tests the functionality of the InferenceGPTNeoX class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Ensures test_questions.json exists and writes it if it does not.
        """
        if not os.path.exists('test_questions.json'):
            write_questions()

    def setUp(self):
        """
        Initializes our llm and set of questions before each test.
        """
        self.llm = InferenceGPTNeoX()
        with open('test_questions.json', 'r') as f:
            questions = json.load(f)
        self.max_question = questions['max_question']
        self.random_questions = questions['random_questions']

    def test_answer(self):
        """
        Ensures the llm can correctly generate an answer for a single question.
        """
        for question in self.random_questions:
            answer = self.llm.answer(question)
            self.assertTrue(isinstance(answer, str))
            self.assertTrue(len(answer) > 0)

    def test_answer_long_question(self):
        """
        Ensures the llm can correctly generate an answer for a single question, which is the longest possible prompt in
        the MS MARCO dataset.
        """
        answer = self.llm.answer(self.max_question)
        self.assertTrue(isinstance(answer, str))
        self.assertTrue(len(answer) > 0)


if __name__ == '__main__':
    unittest.main()
