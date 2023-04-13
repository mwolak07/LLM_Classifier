from typing import List, Optional, Tuple, Union
from collections.abc import Sequence
from torch.utils.data import Dataset
from numpy import ndarray
from src.dataset import LLMClassifierDatabase, MSMarcoDataset
from src.dataset_llms import InferenceLLM


# Storing the type definition for a Feature, to make things simpler.
Feature = Union[str, ndarray[float]]


class LLMClassifierDataset(Sequence, Dataset):
    """
    Responsible for providing a performant and easy-to-use interface for dataset for the GPT LLM classification problem.
    Uses GPTClassifierDatabase internally to store the dataset on disk.

    Attributes:
        _vectorizer: (class attribute) The vectorizer to transform strings into vectors.
        _vectorize: If True, then we apply a vectorizer to every feature string.
        _db: The database containing the gpt classifier dataset.
        _data: A list of Tuples of (feature, label), if we chose to load the entire dataset into memory.
    """
    _vectorizer: None = None
    _vectorize: bool
    _db: LLMClassifierDatabase
    _data: Optional[List[Tuple[Feature, int]]]

    def __init__(self, db_path: str, load_to_memory=False, vectorize=False):
        """
        Initializes the dataset using the database. If load_to_memory is True, it will store the entire dataset in
        memory when the object is initialized.

        Args:
            db_path: The location on disk of the database we are getting our data from.
            load_to_memory: If true, we will load the contents of the database into memory. This will only be beneficial
                            if our dataset is smaller than our RAM.
            vectorize: If true, we will use self.vectorizer to turn the string feature into a vector of floats.
        """
        self._vectorize = vectorize
        self._db = LLMClassifierDatabase(db_path)
        self._data = None
        if load_to_memory:
            self._data = self._load_dataset_to_memory()

    def _load_dataset_to_memory(self) -> List[Tuple[Feature, int]]:
        """
        Loads the dataset into memory by creating a list of GPTClassifierItem, which is the rows in the database
        reformatted to be a list of the human answer followed by the LLM answer with the appropriate label for each
        element.
        """
        # Getting the human and LLM answers with the appropriate labels.
        human_answers = [(row.human_answer, 0) for row in self._db]
        llm_answers = [(row.llm_answer, 1) for row in self._db]
        # Interleaving the answers.
        data = [x for pair in zip(human_answers, llm_answers) for x in pair]
        # Vectorizing if needed.
        if self._vectorize:
            return [self._vectorizer(element[0]) for element in data]
        else:
            return data

    def create_database(self, ms_marco_dataset: MSMarcoDataset, llm: InferenceLLM, short_prompts: bool = False) -> None:
        """
        Creates the database. This uses the MS_Marco dataset along with the llm to:
        - Insert the context and human answers to the database
        - Get the prompts for each element in the database
        - Generate the llm answers using the prompts and the llm
        - Insert the llm answers into the database.

        Args:
            ms_marco_dataset: The MS_Marco dataset we are using to build the gpt classifier dataset.
            llm: The large language model that will be answering the prompts for comparison with human answers.
            short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                           will be excluded. This will remove the "no answer" instruction from the prompt as well.
        """
        # Adding MS Marco to the database.
        self._db.add_ms_marco_dataset(ms_marco_dataset, short_prompts)
        # Getting all of the prompts for the LLM and adding its answers to the database.
        prompts = self._db.prompts()
        llm_answers = llm.answers(prompts)
        self._db.add_llm_answers(llm_answers)

    def __getitem__(self, index: int) -> Tuple[Feature, int]:
        """
        Gets the item at the given index in this dataset. This item is a tuple of the feature and the label.
        - If self._data is not None, we load from self.data. Otherwise, we load from self._db.
        - If self._vectorize is True, we apply self._vectorizer to the feature string.
        When we are loading from self._db, we pull from (index // 2) for even indexes, which will be human answers, and
        (index // 2) + 1 for odd indexes, which will be LLM answers.
        """
        # Getting the element from self._data.
        if self._data is not None:
            feature, label = self._data[index]
        # Getting the element from the database.
        else:
            # Index is even, get the human answer at index // 2.
            if index % 2 == 0:
                feature = self._db[index // 2].human_answer
                label = 0
            # Index is odd, get the LLM answer at index // 2 + 1.
            else:
                feature = self._db[index // 2 + 1].llm_answer
                label = 1
        # Applying vectorization if needed
        feature = self._vectorizer(feature) if self._vectorize else feature
        return feature, label

    def __len__(self) -> int:
        """
        Gets the number of items in this dataset. Since the database stores the human and LLM answers on the same row,
        this will be twice the length of the database.

        Returns:
            The number of elements in this dataset.
        """
        return len(self._db) * 2
