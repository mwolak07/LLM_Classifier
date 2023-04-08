from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import numpy as np
import random
import json


@dataclass
class MSMarcoQueryType(Enum):
    """
    Represents the query types present in the MS MARCO dataset.
    """
    DESCRIPTION = 'DESCRIPTION'
    NUMERIC = 'NUMERIC'
    ENTITY = 'ENTITY'
    PERSON = 'PERSON'
    LOCATION = 'LOCATION'


@dataclass
class MSMarcoItem:
    """
    Represents an element of the MSMarco dataset.

    Attributes:
        query: The query that was asked to the participant.
        query_type: The subject the query was on.
        passages: A list of supporting passages were the answer might be.
        chosen_passages: A list of passages that the participants selected as containing the answer.
        answers: A list of answers given. May be more of there were multiple valid answers given.
    """
    query: str
    query_type: MSMarcoQueryType
    passages: List[str]
    chosen_passages: List[str]
    answers: List[str]


class MSMarcoDataset:
    """
    Responsible for providing an easy-to-use interface for the MS MARCO question-answer dataset.
    Functions as an iterator, with elements of MSMarcoItem.

    Attributes:
        _data_file: The location on disk of the data file.
        _data: Internal list representing a list of elements of the MS Marco dataset.
    """
    _data_file: str
    _data: List[MSMarcoItem]

    def __init__(self, data_file: str):
        """
        Creates a new class with a given data file.
        """
        self._data_file = data_file
        self._data = self._load_data()

    def __getitem__(self, index: int) -> MSMarcoItem:
        """
        Gets the item at the specified index using the '[]' operator.

        Args:
            index: The index of the MS Marco element we are accessing.

        Returns:
            A representation of one element of the MS Marco dataset.
        """
        return self._data[index]

    def __len__(self) -> int:
        """
        Gets the number of elements in the dataset.

        Returns:
            The number of elements in the dataset.
        """
        return len(self._data)

    def sample(self, n: int) -> List[MSMarcoItem]:
        """
        Returns a list of n random elements from the MS Marco dataset.

        Args:
            n: The number of elements to sample.

        Returns:
            A list of n random elements from MS Marco.
        """
        return random.sample(self._data, n)

    def list(self) -> List[MSMarcoItem]:
        """
        Gets this dataset as a list.

        Returns:
            This dataset as a list.
        """
        return self._data.copy()

    def prompt(self, index: int) -> str:
        """
        Generates the language model prompt for the nth element in the dataset. This contains the context of the
        passages, an explanation of how to answer, and the question.

        Args:
            index: The index of the MS Marco element we want to generate the prompt for.

        Returns:
            The LLM prompt for the corresponding element.
        """
        # Getting the nth element in the dataset.
        element = self[index]
        # Providing the model the context passages.
        output = 'With the following passages:\n'
        for passage in element.passages:
            output += passage + '\n\n'
        # Providing the model the query.
        query = element.query
        output += f'Please answer this query: {query}\n'
        # Giving message about no answer.
        no_answer_phrase = 'No Answer Present.'
        output += f'If it is not possible to answer the query using the given prompts, ' \
                  f'please state: {no_answer_phrase}\n'
        return output

    def _load_data(self) -> List[MSMarcoItem]:
        """
        Loads the data from the data_file.

        Returns:
            A list of MSMarcoItem representing the contents of the data_file.
        """
        with open(self._data_file, 'r') as f:
            data = json.load(f)
        queries = self._get_queries(data)
        query_types = self._get_query_types(data)
        passages = self._get_passages(data)
        chosen_passages = self._get_chosen_passages(data)
        answers = self._get_combined_answers(self._get_answers(data), self._get_well_formed_answers(data))
        return [MSMarcoItem(query, query_type, passage_list, chosen_passage_list, answer_list)
                for query, query_type, passage_list, chosen_passage_list, answer_list
                in zip(queries, query_types, passages, chosen_passages, answers)]

    @staticmethod
    def _get_queries(data: Dict[str, Any]) -> List[str]:
        """
        Gets the list of queries from the loaded data.

        Args:
            data: The data loaded from self._data_file.

        Returns:
            A list of queries from the loaded data.
        """
        n = len(data['query'].keys())
        queries = list(np.empty((n,)))
        for key in data['query'].keys():
            i = int(key)
            queries[i] = data['query'][key]
        return queries

    @staticmethod
    def _get_query_types(data: Dict[str, Any]) -> List[MSMarcoQueryType]:
        """
        Loads the list of query types from the loaded data.

        Args:
            data: The data loaded from self._data_file.

        Returns:
            A list of query types from the loaded data, as MSMarcoQueryType.
        """
        n = len(data['query_type'].keys())
        query_types = list(np.empty((n,)))
        for key in data['query_type'].keys():
            i = int(key)
            query_types[i] = MSMarcoQueryType[data['query_type'][key]]
        return query_types

    @staticmethod
    def _get_passages(data: Dict[str, Any]) -> List[List[str]]:
        """
        Loads the list of lists of passages from the loaded data.

        Args:
            data: The data loaded from self._data_file.

        Returns:
            A list of passages from the loaded data.
        """
        n = len(data['passages'].keys())
        passages = list(np.empty((n,)))
        for key in data['passages'].keys():
            i = int(key)
            passages[i] = [passage['passage_text'] for passage in data['passages'][key]]
        return passages

    @staticmethod
    def _get_chosen_passages(data: Dict[str, Any]) -> List[List[str]]:
        """
        Loads the list of lists of passages which were chosen by participants as containing the answer from the loaded
        data.

        Args:
            data: The data loaded from self._data_file.

        Returns:
            A list of passages chosen by participants from the loaded data.
        """
        n = len(data['passages'].keys())
        chosen_passages = list(np.empty((n,)))
        for key in data['passages'].keys():
            i = int(key)
            chosen_passages[i] = [passage['passage_text'] for passage in data['passages'][key]
                                  if passage['is_selected'] == 1]
        return chosen_passages

    @staticmethod
    def _get_answers(data: Dict[str, Any]) -> List[List[str]]:
        """
        Loads the list of answers from the loaded data. Deals with the ['No Answer Present.'] empty case.

        Args:
            data: The data loaded from self._data_file.

        Returns:
            A list of answers from the loaded data.
        """
        n = len(data['answers'].keys())
        answers = list(np.empty((n,)))
        for key in data['answers'].keys():
            i = int(key)
            if data['answers'][key] == ['No Answer Present.']:
                answers[i] = []
            else:
                answers[i] = data['answers'][key]
        return answers

    @staticmethod
    def _get_well_formed_answers(data: Dict[str, Any]) -> List[List[str]]:
        """
        Loads the list of well formed answers from the loaded data. Deals with the '[]' empty case.

        Args:
            data: The data loaded from self._data_file.

        Returns:
            A list of well formed from the loaded data.
        """
        n = len(data['wellFormedAnswers'].keys())
        well_formed_answers = list(np.empty((n,)))
        for key in data['wellFormedAnswers'].keys():
            i = int(key)
            if data['wellFormedAnswers'][key] == '[]':
                well_formed_answers[i] = []
            else:
                well_formed_answers[i] = data['wellFormedAnswers'][key]
        return well_formed_answers

    @staticmethod
    def _get_combined_answers(answers: List[List[str]], well_formed_answers: List[List[str]]) -> List[List[str]]:
        """
        Combines the list of answers and well formed answers into a combined_answers list, where well formed answers
        replace answers.

        Args:
            answers: All of the answers given by participants.
            well_formed_answers: All of the answers from participants fixed manually.

        Returns:
            A combined list, which replaces answers in answers with those from well_formed_answers.
        """
        if len(answers) != len(well_formed_answers):
            raise RuntimeError('Different number of elements for answers and well formed answers!')
        combined_answers = list(np.empty((len(answers),)))
        for i in range(len(answers)):
            # Removing repeats of answers, just to be safe.
            for well_formed_answer in well_formed_answers[i]:
                if well_formed_answer in answers[i]:
                    well_formed_answers[i].remove(well_formed_answer)
            if len(well_formed_answers[i]) > 0:
                combined_answers[i] = well_formed_answers[i]
            else:
                combined_answers[i] = answers[i]
        return combined_answers
