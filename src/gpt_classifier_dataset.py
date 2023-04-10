from dataclasses import dataclass
from numpy import ndarray
from src import GPTClassifierDatabase, GPTClassifierRow


@dataclass
class GPTClassifierItem:
    """
    Represents one piece of vectorized text, human or GPT LLM generated, with a label. 0 for human, 1 for AI.
    """
    text: ndarray[float]
    label: int


class GPTClassifierDataset:
    """
    Responsible for providing a performant and easy-to-use interface for dataset for the GPT LLM classification problem.
    Uses GPTClassifierDatabase internally to store the dataset on disk.
    """
    pass
