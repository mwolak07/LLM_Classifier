from dataclasses import dataclass
from numpy import ndarray
from src import GPTClassifierDatabase, GPTClassifierRow, MSMarcoDataset


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

    Attributes:
        db: The database containing the gpt classifier dataset.
    """
    db: GPTClassifierDatabase
    ms_marco_dataset: MSMarcoDataset

    def __init__(self, db_loc: str, ms_marco_loc: str):

        self.db = GPTClassifierDatabase(db_loc)
        self.ms_marco_dataset = MSMarcoDataset(ms_marco_loc)

    def create_dataset(self):
        """
        Creates the dataset. This uses the MS

        """


