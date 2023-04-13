import unittest
import json
import os
from src.datasets import InferenceGPTNeoX
from src.util import cd_from_root
from test.datasets import write_questions


class TestInferenceGPTNeoX(unittest.TestCase):
    """
    Tests the functionality of the InferenceGPTNeoX class.
    """
    test_questions_path: str = './datasets/test_questions.json'

    @classmethod
    def setUpClass(cls):
        """
        Ensures test_questions.json exists and writes it if it does not.
        """
        # Cd to /test if we are at root.
        cd_from_root('test')
        if not os.path.exists(cls.test_questions_path):
            write_questions()

    def setUp(self):
        """
        Initializes our llm and set of questions before each test.
        """
        self.llm = InferenceGPTNeoX()
        with open(self.test_questions_path, 'r') as f:
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
