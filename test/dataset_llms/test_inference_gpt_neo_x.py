import unittest
from src.dataset_llms import InferenceGPTNeoX
from test.dataset_llms import TestInferenceLLM


class TestInferenceGPTNeoX(TestInferenceLLM):
    """
    Tests the functionality of the InferenceGPTNeoX class. Inherits most of its test from TestInferenceLLM, changing
    only which llm is running using self.llm.

    Attributes:
        test_questions_path: (class attribute) The path to test_questions.json from test root.
        llm: The Inference LLM we are using in this test. To be initialized in setUp in child tests.
        max_question: The max-size question from test_questions.json.
        random_questions: The randomly sampled questions from test_questions.json.
    """

    def setUp(self):
        """
        Initializes the llm, on top of the default behavior in TestInferenceLLM.
        """
        super().setUp()
        self.llm = InferenceGPTNeoX()


if __name__ == '__main__':
    unittest.main()
