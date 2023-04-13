from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, PreTrainedModel, PreTrainedTokenizer
from typing import List
from src.datasets import InferenceLLM


class InferenceGPTNeoX(InferenceLLM):
    """
    Represents an LLM that we can use for inference.

    Attributes:
        _temperature: (class attribute) The temperature for GPT Neo-X.
        _model: The model we will use to perform inference.
        _tokenizer: The model we will use to transform the input strings into vectors.
    """
    _temperature: float = 0.9
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer

    def __init__(self):
        """
        Initializes the model and tokenizer with the appropriate parameters for inference.
        """
        self._model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
        self._tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    def answer(self, question: str) -> str:
        """
        Generates an answer based on the given question.

        Args:
            question: The question GPT Neo-X should respond to.

        Returns:
            The answer given by GPT Neo-X.
        """
        # Tokenizing the input.
        input_token_ids = self._tokenizer(question, return_tensors="pt").input_ids
        # Performing the inference.
        output_tokens = self._model.generate(
            input_token_ids,
            do_sample=True,
            temperature=self._temperature,
            max_length=len(question),
        )
        # Getting text back from the tokenized output.
        answer = self._tokenizer.batch_decode(output_tokens)[0]
        return answer

    def answers(self, questions: List[str]) -> List[str]:
        """
        Generates a list of answers based on the given list of questions.

        Args:
            questions: The list of questions GPT Neo-X should respond to.

        Returns:
            The answers given by GPT Neo-X.
        """
        # Tokenizing the inputs.
        input_token_ids = self._tokenizer(questions, return_tensors='pt').input_ids
        # Performing the batch inference.
        output_tokens = self._model.generate(
            input_token_ids,
            do_sample=True,
            temperature=self._temperature,
            max_length=max([len(question) for question in questions]),
        )
        # Getting text back from the tokenized output.
        answers = self._tokenizer.batch_decode(output_tokens)
        return answers
