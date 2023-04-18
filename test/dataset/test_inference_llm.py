from typing import List
import unittest
import json
import os
from src.util import cd_to_executing_file, get_ram_gb, get_vram_gb
from src.dataset import InferenceLLM
from test.dataset.generate_mock_ms_marco_data import write_mock_ms_marco_data


class TestInferenceLLMUtils(unittest.TestCase):
    """
    Tests the utility functions in the InferenceLLM class.
    """
    test_model_ram_file = './test_model_ram.json'

    def setUp(self):
        """
        Going to the directory of this file before each test.
        """
        # Cd to the directory this file is in.
        cd_to_executing_file(__file__)

    def tearDown(self):
        """
        Removes self.test_model_ram_file after test completes.
        """
        os.remove(self.test_model_ram_file)

    def write_test_model_ram(self, ram: float) -> None:
        """
        Write the test_model_ram file to disk, with less, equal, and greater cases.

        Args:
            ram: The RAM capacity we will be testing for.
        """
        test_model_ram = {
            "greater": ram + 1,
            "equal": ram,
            "less": ram - 1
        }
        with open(self.test_model_ram_file, 'w+') as f:
            json.dump(test_model_ram, f, indent=4)

    def test_check_ram(self):
        """
        Tests that the check_ram function works correctly.
        """
        # Checking for RAM and writing the test RAM file.
        ram = get_ram_gb()
        self.write_test_model_ram(ram)
        # Checking when system RAM < minimum RAM.
        with self.assertRaises(RuntimeError):
            InferenceLLM.check_ram(self.test_model_ram_file, 'greater', False)
        # Checking when system RAM = minimum RAM.
        InferenceLLM.check_ram(self.test_model_ram_file, 'equal', False)
        # Checking when system RAM > minimum RAM.
        InferenceLLM.check_ram(self.test_model_ram_file, 'less', False)
        # Checking for VRAM and writing the test RAM file.
        vram = get_vram_gb()
        self.write_test_model_ram(vram)
        # Checking when system VRAM < minimum RAM.
        with self.assertRaises(RuntimeError):
            InferenceLLM.check_ram(self.test_model_ram_file, 'greater', True)
        # Checking when system VRAM = minimum RAM.
        InferenceLLM.check_ram(self.test_model_ram_file, 'equal', True)
        # Checking when system VRAM > minimum RAM.
        InferenceLLM.check_ram(self.test_model_ram_file, 'less', True)


class TestInferenceLLM(unittest.TestCase):
    """
    Tests the functionality of a InferenceLLM class.

    Attributes:
        test_questions_path: (class attribute) The path to test_questions.json from test root.
        llm: The Inference LLM we are using in this test. To be initialized in setUp in child tests.
        max_question: The max-size question from test_questions.json.
        random_questions: The randomly sampled questions from test_questions.json.
    """
    test_questions_path: str = 'mock_question_data.json'
    llm: InferenceLLM
    max_question: str
    random_questions: List[str]

    # Identifies this test as an abstract test.
    __test__ = False

    @classmethod
    def setUpClass(cls):
        """
        Ensures test_questions.json exists and writes it if it does not.
        """
        # Write questions if needed.
        if not os.path.exists(cls.test_questions_path):
            write_questions()
        # Set up stack trace for CUDA errors.
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def setUp(self):
        """
        Initializes the loads in test_questions.json.
        """
        # Cd to the directory this file is in.
        cd_to_executing_file(__file__)
        self.load_test_questions()
        self.model_names = ['facebook/opt-1.3b', 'bigscience/bloom-1b1']

    def load_test_questions(self) -> None:
        """
        Loads the test questions from test_questions.json.
        """
        with open(self.test_questions_path, 'r') as f:
            questions = json.load(f)
        self.max_question = questions['max_question']
        self.random_questions = questions['random_questions']

    def test_answer(self):
        """
        Ensures the llm can correctly generate an answer for a single question.
        """
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                llm = InferenceLLM(model_name)
                for question in self.random_questions:
                    answer = llm.answer(question, max_answer_len=250)
                    self.assertTrue(isinstance(answer, str))
                    self.assertTrue(len(answer) > 0)

    def test_answer_long_question(self):
        """
        Ensures the llm can correctly generate an answer for a single question, which is the longest possible prompt in
        the MS MARCO dataset.
        """
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                llm = InferenceLLM(model_name)
                answer = llm.answer(self.max_question, max_answer_len=250)
                self.assertTrue(isinstance(answer, str))
                self.assertTrue(len(answer) > 0)

    def test_answers(self):
        """
        Ensures the llm can correctly generate a set of answers for multiple questions.
        """
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                llm = InferenceLLM(model_name)
                answers = llm.answers(self.random_questions, max_answer_len=250)
                self.assertTrue(len(answers) == len(self.random_questions))
                for answer in answers:
                    self.assertTrue(isinstance(answer, str))
                    self.assertTrue(len(answer) > 0)

    def test_answers_long_question(self):
        """
        Ensures the llm can correctly generate a set of answers for multiple questions, if one of the questions is the
        max question.
        """
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                llm = InferenceLLM(model_name)
                answers = llm.answers(self.random_questions + [self.max_question], max_answer_len=250)
                self.assertTrue(len(answers) == len(self.random_questions))
                for answer in answers:
                    self.assertTrue(isinstance(answer, str))
                    self.assertTrue(len(answer) > 0)


if __name__ == '__main__':
    unittest.main()
