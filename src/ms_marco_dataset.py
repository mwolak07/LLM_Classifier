import numpy as np
import random
import json


class MSMarcoDataset:
    """
    Responsible for providing an easy-to-use interface for the MS MARCO question-answer dataset.
    TODO: Add type hints.
    """

    def __init__(self, data_file: str):
        """
        Creates a new class with a given data file.
        """
        self._data_file = data_file
        self._data = self._load_data()

    def __getitem__(self, index):
        """
        Gets the item at the specified index using the '[]' operator.
        """
        return self._data[index]

    def __setitem__(self, index, value):
        """
        Sets the item at the specified index using the '[]' operator.
        """
        self._data[index] = value

    def __len__(self):
        """
        Gets the length of this item.
        """
        return len(self._data)

    def append(self, value):
        """
        Appends a value to this item.
        """
        self._data.append(value)

    def sample(self, n):
        """
        Returns a list of n random elements.
        """
        return random.sample(self._data, n)

    def list(self):
        """
        Returns this set as a list.
        """
        return self._data.copy()

    def prompt(self, n):
        """
        Generates the language model prompt for the nth element in the dataset. This contains the context of the
        passages, an explanation of how to answer, and the question.
        """
        # Getting the nth element in the dataset.
        element = self[n]
        # Providing the model the context passages.
        output = 'With the following passages:\n'
        for passage in element['passages']:
            output += passage + '\n\n'
        # Providing the model the query.
        query = element['query']
        output += f'Please answer this query: {query}\n'
        # Giving message about no answer.
        no_answer_phrase = 'No Answer Present.'
        output += f'If it is not possible to answer the query using the given prompts, ' \
                  f'please state: {no_answer_phrase}\n'
        return output

    def _load_data(self):
        """
        Loads the data from the data_file.
        """
        with open(self._data_file, 'r') as f:
            data = json.load(f)
        queries = self._get_queries(data)
        query_types = self._get_query_types(data)
        passages = self._get_passages(data)
        answers = self._get_combined_answers(self._get_answers(data), self._get_well_formed_answers(data))
        return [{'query': query, 'query_type': query_type, 'passages': passage_list, 'answers': answer_list}
                for query, query_type, passage_list, answer_list
                in zip(queries, query_types, passages, answers)]

    @staticmethod
    def _get_queries(data):
        """
        Gets the list of queries from the loaded data.
        """
        n = len(data['query'].keys())
        queries = list(np.empty((n,)))
        for key in data['query'].keys():
            i = int(key)
            queries[i] = data['query'][key]
        return queries

    @staticmethod
    def _get_query_types(data):
        """
        Gets the list of query types from the loaded data.
        """
        n = len(data['query_type'].keys())
        query_types = list(np.empty((n,)))
        for key in data['query_type'].keys():
            i = int(key)
            query_types[i] = data['query_type'][key]
        return query_types

    @staticmethod
    def _get_passages(data):
        """
        Gets the list of lists of passages from the loaded data.
        """
        n = len(data['passages'].keys())
        passages = list(np.empty((n,)))
        for key in data['passages'].keys():
            i = int(key)
            passages[i] = [passage['passage_text'] for passage in data['passages'][key]]
        return passages

    @staticmethod
    def _get_answers(data):
        """
        Gets the list of answers from the loaded data. Deals with the ['No Answer Present.'] empty case.
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
    def _get_well_formed_answers(data):
        """
        Gets the list of well formed answers from the loaded data. Deals with the '[]' empty case.
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
    def _get_combined_answers(answers, well_formed_answers):
        """
        Combines the list of answers and well formed answers into a combined_answers list, where well formed answers
        replace answers.
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
