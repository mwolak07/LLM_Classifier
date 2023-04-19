from typing import List, Optional, Tuple, Union
from collections.abc import Sequence
from torch.utils.data import Dataset
from numpy import ndarray
from torch import cuda
import statistics
import time
from src.dataset import LLMClassifierDatabase, InferenceLLM, MSMarcoDataset


# Storing the type definition for a Feature, to make things simpler.
Feature = Union[str, ndarray[float]]


class LLMClassifierDataset(Sequence, Dataset):
    """
    Responsible for providing a performant and easy-to-use interface for dataset for the GPT LLM classification problem.
    Uses GPTClassifierDatabase internally to store the dataset on disk.

    Attributes:
        _db: The database containing the gpt classifier dataset.
        _data: A list of Tuples of (feature, label), if we chose to load the entire dataset into memory.
    """
    prompt_into: str = 'Using only the following context:'
    prompt_question: str = 'The short answer, in complete sentences, to the question:'
    prompt_end: str = ', is:'
    _db: LLMClassifierDatabase
    _data: Optional[List[Tuple[Feature, int]]]

    def __init__(self, db_path: str, load_to_memory: bool = False):
        """
        Initializes the dataset using the database. If load_to_memory is True, it will store the entire dataset in
        memory when the object is initialized.

        Args:
            db_path: The location on disk of the database we are getting our data from.
            load_to_memory: If true, we will load the contents of the database into memory. This will only be beneficial
                            if our dataset is smaller than our RAM.
        """
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
        return data

    def __getitem__(self, index: int) -> Tuple[Feature, int]:
        """
        Gets the item at the given index in this dataset. This item is a tuple of the feature and the label.
        - If self._data is not None, we load from self.data. Otherwise, we load from self._db.
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
        return feature, label

    def __len__(self) -> int:
        """
        Gets the number of items in this dataset. Since the database stores the human and LLM answers on the same row,
        this will be twice the length of the database.

        Returns:
            The number of elements in this dataset.
        """
        return len(self._db) * 2

    def tolist(self) -> List[Tuple[Feature, int]]:
        """
        Turns this dataset into a list of Tuples of feature: label.

        Returns:
            This dataset as a list.
        """
        if self._data is not None:
            return self._data.copy()
        else:
            return self._load_dataset_to_memory()

    def create_database(self, ms_marco_dataset: MSMarcoDataset, llm: InferenceLLM, short_prompts: bool = True,
                        batch_size: int = 1) -> None:
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
            batch_size: The batch size that can be optionally specified for inference, if you have a lot of RAM.
        """
        # Adding MS Marco to the database, and clearing it from memory when we are done.
        print('Inserting MS MARCO into the database...')
        self._db.add_ms_marco_dataset(ms_marco_dataset, short_prompts)
        del ms_marco_dataset
        # Adding the prompts to the database, and clearing them from memory when we are done.
        print('Getting the database rows...')
        rows = self._db.tolist()
        print('Inserting prompts into the database...')
        prompts = [self.prompt(row.passages, row.query) for row in rows]
        self._db.add_prompts(prompts)
        del prompts
        # Adding the LLM answers to the database. We don't use the more convenient answers(), because if we stop midway,
        # he database would not be populated.
        print('Inserting LLM answers into the database...')
        answer_lengths = [len(answer) for answer in self._db.human_answers()]
        max_answer_len = int(round(statistics.mean(answer_lengths) + (statistics.stdev(answer_lengths) * 3)))
        self.llm_answers_to_db(llm=llm, max_answer_len=max_answer_len, prompts=self._db.prompts(),
                               batch_size=batch_size, start_answer_index=0, start_batch_index=0)

    @staticmethod
    def prompt(passages: List[str], query: str) -> str:
        """
        Generates the language model prompt for the nth element in the dataset. This contains the context of the
        passages, an explanation of how to answer, and the question. The context can be all of the passages, or only
        the ones chosen as relevant by the human respondent. Note, that there will be no chosen passages if the human
        respondent stated that there were no answers.

        Assume:
            If short is True, then answers is not empty.

        Args:
            passages: The passages to use as context for the query.
            query: The query the prompt should be asking about.

        Returns:
            The LLM prompt for the corresponding element.
        """
        # Remove the question mark from the end of the query, if it is there.
        if query[-1] == '?':
            query = query[:-1]
        # Providing the model the context passages.
        output = f'Using only the following context:\n'
        for passage in passages:
            output += passage + '\n\n'
        # Providing the model the query.
        output += f'The short answer, in complete sentences, to the question: "{query}?", is:\n'
        return output

    def llm_answers_to_db(self, llm: InferenceLLM, max_answer_len: int, prompts: List[str], batch_size: int = 1,
                          start_answer_index: int = 0, start_batch_index: int = 0) -> None:
        """
        Adds answers to the database from the LLM using batch inference. We do it in a custom loop like this, because
        then we are able to store results in the database at each step, and so we can ensure interrupted inference or
        a system crash will not result in us losing our progress.

        Args:
            llm: The InferenceLLM to use for the inference to generate answers.
            max_answer_len: The maximum length for an answer from the LLM.
            prompts: The list of prompts to generate responses for.
            batch_size: The size of the batches to run the inference with.
            start_answer_index: The index to start inserting answers at (default is 0, but can be higher if we are
                                resuming).
            start_batch_index: The index to answering question batches at (default is 0, but can be higher if we are
                                resuming).
        """
        answer_index = start_answer_index
        prompt_batches = llm.get_batches(prompts, batch_size)
        print(f'Generating answers in {len(prompt_batches)} batches of size {batch_size}...')
        for i in range(start_batch_index, len(prompt_batches)):
            t = time.time()
            prompt_batch = prompt_batches[i]
            answer_batch = [''] * len(prompt_batch)
            print(f'Generating batch {i + 1}/{len(prompt_batches)} of size {len(prompt_batch)}')
            try:
                answer_batch, tries = llm.answer_batch(prompt_batch, answer_batch, max_answer_len)
                print(f'Done in {time.time() - t}s {(i + 1) / len(prompt_batches) * 100}%)')
            except (cuda.CudaError, cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
                print(f'WARNING! Could not generate answers for batch {i}: {str(e)}')
            for answer in answer_batch:
                self._db.add_llm_answer(answer, answer_index)
                answer_index += 1
