from abc import ABC, abstractmethod
from typing import List
import unittest
import json
import os
from src.util import cd_to_executing_file, get_ram_gb, get_vram_gb
from src.dataset import InferenceLLM, MSMarcoDataset, LLMClassifierDataset
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
        if os.path.exists(self.test_model_ram_file):
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

    def test_get_sentences(self):
        """
        Tests that the get_sentences function works correctly to split blocks of text into sentences based on
        punctuation.
        """
        # Testing empty string.
        sentence = ''
        answer = []
        self.assertEqual(answer, InferenceLLM.get_sentences(sentence))
        # Test ending with no punctuation mark.
        sentence = 'Test'
        answer = ['Test']
        self.assertEqual(answer, InferenceLLM.get_sentences(sentence))
        # Testing ending with one punctuation mark.
        for p in ['.', '!', '?']:
            sentence = f'Test{p}'
            answer = [f'Test{p}']
            self.assertEqual(answer, InferenceLLM.get_sentences(sentence))
        # Testing ending with double punctuation marks.
        for p in ['.', '!', '?']:
            sentence = f'Test{p}{p}'
            answer = [f'Test{p}', f'{p}']
            self.assertEqual(answer, InferenceLLM.get_sentences(sentence))
        # Testing starting with punctuation marks.
        for p in ['.', '!', '?']:
            sentence = f'{p}Test'
            answer = [f'{p}', f'Test']
            self.assertEqual(answer, InferenceLLM.get_sentences(sentence))
        # Testing complex mixed punctuation.
        sentence = 'Hello! This is a test. We will see, if this works! The big question is, can it do multiple ' \
                   'punctuation? I guess when this runs, we will find out.'
        answer = ['Hello!', 'This is a test.', 'We will see, if this works!',
                  'The big question is, can it do multiple punctuation?', 'I guess when this runs, we will find out.']
        self.assertEqual(answer, InferenceLLM.get_sentences(sentence))
        # Testing complex mixed punctuation, that cuts off.
        sentence = 'Hello! This is a test. We will see, if this works! The big question is, can it do multiple ' \
                   'punctuation? I guess when this runs, we will find out. It might cut o'
        answer = ['Hello!', 'This is a test.', 'We will see, if this works!',
                  'The big question is, can it do multiple punctuation?', 'I guess when this runs, we will find out.',
                  'It might cut o']
        self.assertEqual(answer, InferenceLLM.get_sentences(sentence))

    def test_postprocess_answer(self):
        """
        Ensures the postprocess_answer method works correctly to remove incomplete sentences and repeats.
        """
        context = 'The answer, in complete sentences, to the question: "question", is:'
        # Testing None.
        sentence = None
        answer = ''
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing empty string.
        sentence = ''
        answer = ''
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing no response.
        sentence = context + ''
        answer = ''
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Test ending with no punctuation mark.
        sentence = context + 'Test'
        answer = 'Test.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Test ending with leading space.
        sentence = context + ' Test.'
        answer = 'Test.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing ending with one punctuation mark.
        for p in ['.', '!', '?']:
            sentence = context + f'Test{p}'
            answer = f'Test{p}'
            self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing ending with double punctuation marks.
        for p in ['.', '!', '?']:
            sentence = context + f'Test{p}{p}'
            answer = f'Test{p}{p}'
            self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing ending with five punctuation marks.
        for p in ['.', '!', '?']:
            sentence = context + f'Test{p}{p}{p}{p}{p}'
            answer = f'Test{p}{p}{p}{p}{p}'
            self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing starting with punctuation marks.
        for p in ['.', '!', '?']:
            sentence = context + f'{p}Test'
            answer = f'Test.'
            self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing a long sentence that ends with punctuation.
        sentence = context + 'Hello! This is a test. We will see, if this works! The big question is, can it do ' \
                             'multiple punctuation? I guess when this runs, we will find out.'
        answer = 'Hello! This is a test. We will see, if this works! The big question is, can it do multiple ' \
                 'punctuation? I guess when this runs, we will find out.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing a long sentence that cuts off.
        sentence = context + 'Hello! This is a test. We will see, if this works! The big question is, can it do ' \
                             'multiple punctuation? I guess when this runs, we will find out. It might cut o'
        answer = 'Hello! This is a test. We will see, if this works! The big question is, can it do multiple ' \
                 'punctuation? I guess when this runs, we will find out.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing a twice repeated sentence.
        sentence = context + 'Hello this is me! Hello this is me!'
        answer = 'Hello this is me!'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing a three times repeated sentence.
        sentence = context + 'Hello this is me. Hello this is me. Hello this is me.'
        answer = 'Hello this is me.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing a non sequential repetition.
        sentence = context + 'Hello this is me. Hello this is not me. Hello this is me.'
        answer = 'Hello this is me. Hello this is not me. Hello this is me.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing repetition after a few sentences.
        sentence = context + 'Hello this is me. I am a robot. I give responses! I give responses!'
        answer = 'Hello this is me. I am a robot. I give responses!'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing repetition in the middle of a few sentences.
        sentence = context + 'Hello this is me. I give responses! I give responses! I am a robot.'
        answer = 'Hello this is me. I give responses! I am a robot.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing repetition after a few sentences, that gets cut off.
        sentence = context + 'Hello this is me. I am a robot. I give responses! I give responses! I give'
        answer = 'Hello this is me. I am a robot. I give responses!'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))
        # Testing repetition in the middle of a few sentences, and the end gets cut off.
        sentence = context + 'Hello this is me. I give responses! I give responses! I am a robot. I like'
        answer = 'Hello this is me. I give responses! I am a robot.'
        self.assertEqual(answer, InferenceLLM.postprocess_answer(sentence))


class TestInferenceLLM(ABC):
    """
    Tests the functionality of a InferenceLLM class. Acts as a base class for sub-classes, which will implement
    model_name with the appropriate model and run the test class for a specific model. This is more efficient, as we can
    then initialize the model and put it on the GPU once in setUpClass, instead of in every method.

    Attributes:
        test_questions_path: (class attribute) The path to test_questions.json from test root.
        llm: The Inference LLM we are using in this test. To be initialized in setUp in child tests.
        max_prompt: The max-size prompt from the mock MS MARCO dataset.
        random_prompts: The random prompts from the mock MS MARCO dataset.
    """
    mock_ms_marco_path: str = 'mock_ms_marco_data.json'
    llm: InferenceLLM
    max_prompt: str
    random_questions: List[str]

    @classmethod
    def setUpClass(cls):
        """
        Ensures test_questions.json exists and writes it if it does not.
        """
        # Write questions if needed.
        if not os.path.exists(cls.mock_ms_marco_path):
            write_mock_ms_marco_data()
        # Setting up the llm.
        cls.llm = InferenceLLM(cls.model_name())

    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        """
        To be implemented by each subclass.

        Returns:
            The model name for this test class.
        """
        pass

    def setUp(self):
        """
        Initializes the loads in test_questions.json.
        """
        # Cd to the directory this file is in.
        cd_to_executing_file(__file__)
        self.load_test_questions()

    def load_test_questions(self) -> None:
        """
        Loads the test questions from test_questions.json.
        """
        # Reading the mock dataset in.
        dataset = MSMarcoDataset(self.mock_ms_marco_path)
        # Getting the longest long prompt.
        long_prompts = [LLMClassifierDataset.prompt(element.passages, element.query) for element in dataset]
        self.max_prompt = max(long_prompts, key=len)
        # Storing the short prompts.
        short_prompts = [LLMClassifierDataset.prompt(element.chosen_passages, element.query) for element in dataset]
        self.random_prompts = short_prompts

    def test_answer(self):
        """
        Ensures the llm can correctly generate an answer for a number of single short questions.
        """
        # Going through our set of random prompts.
        for question in self.random_prompts:
            answer = self.llm.answer(question, max_answer_len=250)
            self.assertTrue(isinstance(answer, str))
            self.assertTrue(len(answer) > 0)

    def test_answer_long_question(self):
        """
        Ensures the llm can correctly generate an answer for a single question, which is the longest possible prompt in
        the MS MARCO dataset.
        """
        answer = self.llm.answer(self.max_prompt, max_answer_len=250)
        self.assertTrue(isinstance(answer, str))
        self.assertTrue(len(answer) > 0)

    def test_answers_batch_n(self):
        """
        Ensures the llm can correctly generate a set of answers for multiple short questions, with various batch sizes.
        """
        for batch_size in [2, 4, 8, 16, 32]:
            with self.subTest(batch_size=batch_size):
                answers = self.llm.answers(self.random_prompts, max_answer_len=250, batch_size=batch_size)
                self.assertTrue(len(answers) == len(self.random_prompts))
                for answer in answers:
                    self.assertTrue(isinstance(answer, str))
                    self.assertTrue(len(answer) > 0)

    def test_answers_long_question_batch_n(self):
        """
        Ensures the llm can correctly generate a set of answers for multiple questions, if one of the questions is the
        max question, with various batch sizes.
        """
        for batch_size in [2, 4, 8, 16, 32]:
            with self.subTest(batch_size=batch_size):
                answers = self.llm.answers(self.random_prompts + [self.max_prompt], max_answer_len=250,
                                           batch_size=batch_size)
                self.assertTrue(len(answers) == len(self.random_prompts) + 1)
                for answer in answers:
                    self.assertTrue(isinstance(answer, str))
                    self.assertTrue(len(answer) > 0)


class TestBloom11B(TestInferenceLLM, unittest.TestCase):
    """
    Runs TestInferenceLLM for 1.1B parameter BLOOM model.
    """

    @classmethod
    def model_name(cls) -> str:
        """
        Implements the Bloom 1.3B model name.

        Returns:
            The model name for this test class.
        """
        return 'bigscience/bloom-1b1'


class TestOPT13B(TestInferenceLLM, unittest.TestCase):
    """
    Runs TestInferenceLLM for 1.3B parameter OPT model.
    """

    @classmethod
    def model_name(cls) -> str:
        """
        Implements the OPT 1.3B model name.

        Returns:
            The model name for this test class.
        """
        return 'facebook/opt-1.3b'


class TestGPTNeo13B(TestInferenceLLM, unittest.TestCase):
    """
    Runs TestInferenceLLM for 1.3B parameter GPT-Neo model.
    """

    @classmethod
    def model_name(cls) -> str:
        """
        Implements the GPT-Neo 1.3B model name.

        Returns:
            The model name for this test class.
        """
        return 'EleutherAI/gpt-neo-1.3B'


if __name__ == '__main__':
    unittest.main()
