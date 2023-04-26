from gensim.models.fasttext import load_facebook_model, FastText
from typing import List, Optional, Tuple, Any
from collections.abc import Sequence
from numpy import ndarray
from torch import cuda
from tqdm import tqdm
import numpy as np
import time
from src.dataset import LLMClassifierDatabase, LLMClassifierRow, InferenceLLM, MSMarcoDataset
from src.util import Feature, cd_to_executing_file, get_batches


class LLMClassifierDataset(Sequence):
    """
    Responsible for providing a performant and easy-to-use interface for dataset for the GPT LLM classification problem.
    Uses GPTClassifierDatabase internally to store the dataset on disk.

    Attributes:
        prompt_into: (class attribute) The first part of the prompt, setting up the context.
        prompt_question: (class attribute) The part of the prompt setting up the question.
        prompt_end: (class attribute) The last part of the prompt, encouraging an answer
        _fasttext: If this is true, then we use the fasttext vectorized versions of string data points.
        _db: The database containing the gpt classifier dataset.
        _data: A list of Tuples of (feature, label), if we chose to load the entire dataset into memory.
    """
    prompt_into: str = 'Using only the following context:'
    prompt_question: str = 'The short answer, in complete sentences, to the question:'
    prompt_end: str = ', is:'
    _fasttext: bool
    _db: LLMClassifierDatabase
    _data: Optional[List[Tuple[Feature, int]]]

    def __init__(self, db_path: str, fasttext: bool = False, load_to_memory: bool = False):
        """
        Initializes the dataset using the database. If load_to_memory is True, it will store the entire dataset in
        memory when the object is initialized.

        Args:
            db_path: The location on disk of the database we are getting our data from.
            fasttext: If true, we use the fasttext vectorized versions of the features, instead of the strings.
            load_to_memory: If true, we will load the contents of the database into memory. This will only be beneficial
                            if our dataset is smaller than our RAM.
        """
        self._fasttext = fasttext
        self._db = LLMClassifierDatabase(db_path)
        self._data = None
        if load_to_memory:
            self._data = self._load_dataset_to_memory()

    def _load_dataset_to_memory(self) -> List[Tuple[Feature, int]]:
        """
        Loads the dataset into memory by creating a list of GPTClassifierItem, which is the rows in the database
        reformatted to be a list of the human answer followed by the LLM answer with the appropriate label for each
        element.

        Returns:
            A list representation of this dataset, which is the human and llm answers interleaved.
        """
        # Getting the human and LLM answers with the appropriate labels.
        if self._fasttext:
            human_answers = [(row.human_answer_fasttext, 0) for row in self._db]
            llm_answers = [(row.llm_answer_fasttext, 1) for row in self._db]
        else:
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
        # Getting the element at index // 2 from the database.
        else:
            feature_element = self._db[index // 2]
            # Index is even, get the human answer.
            if index % 2 == 0:
                feature = feature_element.human_answer_fasttext if self._fasttext else feature_element.human_answer
                label = 0
            # Index is odd, get the llm answer.
            else:
                feature = feature_element.llm_answer_fasttext if self._fasttext else feature_element.llm_answer
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

    def max_words(self) -> int:
        """
        Gets the maximum number of words in both the human and llm answers.

        Returns:
            The maximum number of words in an answer.
        """
        max_human_answer = max([len(human_answer.split(' ')) for human_answer in self._db.human_answers()])
        max_llm_answer = max([len(llm_answer.split(' ')) for llm_answer in self._db.llm_answers()])
        return max(max_human_answer, max_llm_answer)

    def create_database(self, ms_marco_dataset: MSMarcoDataset, llm_name: str, vectorizer_path: str,
                        short_prompts: bool = True, batch_size: int = 1) -> None:
        """
        Creates the database. This uses the MS_Marco dataset along with the llm to:
        - Insert the context and human answers to the database
        - Get the prompts for each element in the database
        - Generate the llm answers using the prompts and the llm
        - Insert the llm answers into the database.

        Args:
            ms_marco_dataset: The MS_Marco dataset we are using to build the gpt classifier dataset.
            llm_name: The name of the large language model that will be answering the prompts for comparison with human
                      answers.
            vectorizer_path: The path to the word vectorizer to apply to the human and llm answers after we have them.
            short_prompts: If this is True, we will use only the chosen passages in the prompts, and "no answer" cases
                           will be excluded. This will remove the "no answer" instruction from the prompt as well.
            batch_size: The batch size that can be optionally specified for inference, if you have a lot of RAM.
        """
        # Adding MS Marco to the database, and clearing it from memory when we are done.
        print(f'Inserting MS MARCO into the database...')
        self._db.add_ms_marco_dataset(ms_marco_dataset, short_prompts)
        del ms_marco_dataset

        # Getting the prompts and answer lengths for the LLM, and clearing the rows from memory when we are done.
        print(f'Getting the database rows...')
        rows = self._db.tolist()
        print(f'Generating prompts and target answer lengths...')
        prompts, answer_lengths, sorted_indices = self.generate_llm_prompts(rows)
        del rows

        # Adding the prompts and LLM answers to the database. We don't use the more convenient answers(), because if we
        # stop midway, the database would not be populated. Note this insertion will be out of order until it is fixed.
        print(f'WARNING: The prompts and answers will now be inserted in the incorrect order. If the program crashes '
              f'before it fixes this order, the following indices from the sort by length must be used to fix it: '
              f'{sorted_indices}')
        print(f'Inserting prompts into the database...')
        self._db.add_prompts(prompts)
        print(f'Loading LLM into memory...')
        llm = InferenceLLM(llm_name)
        print(f'Inserting LLM answers into the database...')
        self.llm_answers_to_db(llm=llm, prompts=prompts, answer_lengths=answer_lengths, batch_size=batch_size,
                               start_answer_index=0, start_batch_index=0)
        del llm, prompts, answer_lengths

        # Fixing the order of the prompts and LLM answers, by getting them from the DB, unsorting them, and updating
        # the DB.
        print(f'Fixing the order of LLM answers in the database...')
        prompts = self.unsort_array(self._db.prompts(), sorted_indices)
        self._db.add_prompts(prompts)
        print(f'Fixing the order of prompts in the database...')
        llm_answers = self.unsort_array(self._db.llm_answers(), sorted_indices)
        self._db.add_llm_answers(llm_answers)
        del prompts, llm_answers

        # Vectorizing the human and llm answers with fasttext, and inserting the results in to the database.
        print(f'Loading vectorizer into memory...')
        vectorizer = load_facebook_model(vectorizer_path)
        print(f'Inserting vectorized human and llm answers into the database...')
        rows = self._db.tolist()
        for i in tqdm(range(len(rows))):
            row = rows[i]
            human_answer = self.fasttext_vectorize(row.human_answer, vectorizer)
            llm_answer = self.fasttext_vectorize(row.llm_answer, vectorizer)
            self._db.add_fasttext(human_answer, llm_answer, i)
        del vectorizer, rows

    def generate_llm_prompts(self, rows: List[LLMClassifierRow]) -> Tuple[List[str], List[int], List[int]]:
        """
        Generates the prompts for the LLM, the lengths for the human answers associated with each prompt, and the
        sorted indices resulting from sorting the prompts and human answers. This is done so that when we batch prompts
        for the LLM, we can supply a target length that is similar to the lengths of the answers given by the humans.
        The indices are needed to "unsort" the resulting LLM answers for insertion into the database.

        Args:
            rows: A list of the rows from the database, which contain all of the MS Marco and human response data to
                  build the prompts for the LLM.

        Returns:
            The sorted prompts for the LLM, the sorted target answer lengths for each prompt, and the sorted indices.
        """
        # Getting the prompts and human answer lengths from the rows.
        prompts = [self.prompt(row.passages, row.query) for row in rows]
        answer_lengths = [len(row.human_answer) for row in rows]
        # Sorting the answer_lengths array, and getting the sorted indices.
        answer_lengths, sorted_indices = self.sort_array_value(answer_lengths)
        # "Sorting" the prompts to the same order as the answer_lengths array.
        prompts = self.sort_array_indices(prompts, sorted_indices)
        return prompts, answer_lengths, sorted_indices

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
        if len(query) > 0 and query[-1] == '?':
            query = query[:-1]
        # Providing the model the context passages.
        output = f'Using only the following context:\n'
        for passage in passages:
            output += passage + '\n\n'
        # Providing the model the query.
        output += f'The short answer, in complete sentences, to the question: "{query}?", is:\n'
        return output

    @staticmethod
    def sort_array_value(array: List[Any]) -> Tuple[List[Any], List[int]]:
        """
        Sorts the array by value, and returns a list of the sorted indices. This allows the list to be "unsorted"
        after, and other arrays to be "sorted" in the same order.

        Args:
            array: The array to be sorted.

        Returns:
            The sorted array, and the sorted indexes.

        Raises:
            RuntimeError: If argsort returns an int instead of an ndarray.
        """
        # Using numpy argsort to accomplish this operation.
        array = np.array(array)
        sorted_indices = np.argsort(array)
        # Checking the type of sorted_indices
        if isinstance(sorted_indices, int):
            raise RuntimeError('Numpy argsort returned only an int!')
        sorted_indices = np.array(sorted_indices)
        array = array[sorted_indices]
        # Converting to python lists and returning.
        return array.tolist(), sorted_indices.tolist()

    @staticmethod
    def sort_array_indices(array: List[Any], sorted_indices: List[int]) -> List[Any]:
        """
        Sorts the given array according to a pre-determined order, given by the sorted indices.

        Args:
            array: The array to be sorted.
            sorted_indices: The sorted indices encoding the order we want to sort the array by.

        Returns:
            The array sorted in the order supplied by sorted_indices.
        """
        # Using advanced numpy indexing.
        array = np.array(array)
        sorted_indices = np.array(sorted_indices)
        array = array[sorted_indices]
        return array.tolist()

    @staticmethod
    def unsort_array(sorted_array: List[Any], sorted_indices: List[int]) -> List[Any]:
        """
        Unsorted the array, given the sorted array and the sorted indices, using the order encoded in the sorted indices
        to undo the sort.

        Args:
            sorted_array: The array that has already been sorted.
            sorted_indices: The indices encoding the order of the sorted array.

        Returns:
            The sorted_array, back in its original order.
        """
        # Using numpy argsort to accomplish this operation.
        sorted_array = np.array(sorted_array)
        sorted_indices = np.array(sorted_indices)
        array = sorted_array[np.argsort(sorted_indices)]
        return array.tolist()

    def llm_answers_to_db(self, llm: InferenceLLM, prompts: List[str], answer_lengths: List[int], batch_size: int = 1,
                          start_answer_index: int = 0, start_batch_index: int = 0) -> None:
        """
        Adds answers to the database from the LLM using batch inference. We do it in a custom loop like this, because
        then we are able to store results in the database at each step, and so we can ensure interrupted inference or
        a system crash will not result in us losing our progress. Additionally, we sort the batches by length first,


        Args:
            llm: The InferenceLLM to use for the inference to generate answers.
            prompts: The list of prompts to generate responses for.
            answer_lengths: The length of the human answer corresponding to each prompt.
            batch_size: The size of the batches to run the inference with.
            start_answer_index: The index to start inserting answers at (default is 0, but can be higher if we are
                                resuming).
            start_batch_index: The index to answering question batches at (default is 0, but can be higher if we are
                                resuming).
        """
        # This represents 3 extra words of "wiggle room" for the LLM to finish sentences, so we don't end up with output
        # that is on average shorter than the human responses, due to partial end sentences being removed.
        answer_length_pad = 15
        answer_index = start_answer_index
        # Getting the prompts and target answers lengths into batches.
        prompt_batches = get_batches(prompts, batch_size)
        answer_length_batches = get_batches(answer_lengths, batch_size)
        print(f'Generating answers in {len(prompt_batches)} batches of size {batch_size}...')
        # Giving the batch of prompts to the LLM, with the target answer length determined by the answer_length_batches
        # array.
        for i in range(start_batch_index, len(prompt_batches)):
            t = time.time()
            prompt_batch = prompt_batches[i]
            answer_length_batch = answer_length_batches[i]
            answer_batch = [''] * len(prompt_batch)
            print(f'Generating batch {i + 1}/{len(prompt_batches)} of size {len(prompt_batch)}')
            # Guarding against out of memory errors and other crashes when generating with the LLM.
            try:
                max_answer_len = max(answer_length_batch) + answer_length_pad
                answer_batch, tries = llm.answer_batch(prompt_batch, answer_batch, max_answer_len)
                print(f'Done in {time.time() - t}s {(i + 1) / len(prompt_batches) * 100}%)')
            except (cuda.CudaError, cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
                print(f'WARNING! Could not generate answers for batch {i}: {str(e)}')
            # Adding the answers directly into the database, in case the program is interrupted for any reason.
            for answer in answer_batch:
                self._db.add_llm_answer(answer, answer_index)
                answer_index += 1

    @staticmethod
    def fasttext_vectorize(text: str, vectorizer: FastText) -> ndarray[ndarray[np.float32]]:
        """
        Applies the pre-trained fasttext vectorizer as efficiently as possible to the given list of strings.

        Args:
            text: The block of text to be transformed into lists of vectors using the fasttext vectorizer.
            vectorizer: The vectorizer to apply to the text_list.

        Returns:
            The array of text blocks transformed into vector representations.
        """
        output = []
        words = text.split(' ')
        for word in words:
            output.append(vectorizer.wv[word])
        return np.array(output, dtype=np.float32)


def load_data(train_db_path: str, test_db_path: str, fasttext: bool = False) -> \
        Tuple[ndarray[Feature], ndarray[Feature], ndarray[int], ndarray[int]]:
    """
    Loads the data from the LLMClassifierDataset and separates the data into features and labels.

    Args:
        train_db_path: The path the the database containing the training data.
        test_db_path: The path to the database containing the testing data.
        fasttext: Should we use the fasttext vectorized representations instead of strings?

    Returns:
        x_train, x_test, y_train, y_test.
    """
    # Loading in and processing the data.
    cd_to_executing_file(__file__)
    train_dataset = LLMClassifierDataset(train_db_path, fasttext=fasttext, load_to_memory=True)
    test_dataset = LLMClassifierDataset(test_db_path, fasttext=fasttext, load_to_memory=True)
    # Separating our the features and labels.
    x_train = np.array([item[0] for item in train_dataset])
    x_test = np.array([item[0] for item in test_dataset])
    y_train = np.array([item[1] for item in train_dataset])
    y_test = np.array([item[1] for item in test_dataset])
    # Returning the results.
    return x_train, x_test, y_train, y_test
