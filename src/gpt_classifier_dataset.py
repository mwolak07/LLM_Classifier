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

    def __init__(self, db_loc: str):
        """
        Initializes the dataset using the database.
        """
        self.db = GPTClassifierDatabase(db_loc)

    def create_dataset(self, ms_marco_dataset: MSMarcoDataset, llm: None):
        """
        Creates the dataset. This uses the MS_Marco dataset along with the llm to:
        - Insert the context and human answers to the database
        - Get the prompts for each element in the database
        - Generate the llm answers using the prompts and the llm
        - Insert the llm answers into the database.

        Args:
            ms_marco_dataset: The MS_Marco dataset corresponding to
        """
        self.db.add_ms_marco_dataset(ms_marco_dataset)



