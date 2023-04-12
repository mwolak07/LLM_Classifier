from abc import ABC, abstractmethod
from typing import List


class InferenceLLM(ABC):
    """
    Abstract class, representing an interface for LLMs which we can use for inference when generating the dataset.
    """

    @abstractmethod
    def answer(self, question: str) -> str:
        """
        Generates an answer based on the given question.

        Args:
            question: The question the LLM should respond to.

        Returns:
            The answer given by the LLM.
        """
        raise NotImplementedError('Answer not implemented by child class!')

    def answers(self, questions: List[str]) -> List[str]:
        """
        Generates a list of answers based on the given list of questions.

        Args:
            questions: The list of questions the LLM should respond to.

        Returns:
            The answers given by the LLM.
        """
        raise NotImplementedError('Answer not implemented by child class!')
