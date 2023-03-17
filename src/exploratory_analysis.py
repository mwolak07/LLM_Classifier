from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import json


"""
This file is for exploratory analysis of the MS MARCO dataset, with various techniques.
"""


class DataInterface:
    """
    Responsible for providing an easy-to-use interface for the MS MARCO question-answer dataset.
    """

    def __init__(self, data_file):
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
        
    def _get_queries(self, data):
        """
        Gets the list of queries from the loaded data.
        """
        n = len(data['query'].keys())
        queries = list(np.empty((n,)))
        for key in data['query'].keys():
            i = int(key)
            queries[i] = data['query'][key]
        return queries
    
    def _get_query_types(self, data):
        """
        Gets the list of query types from the loaded data.
        """
        n = len(data['query_type'].keys())
        query_types = list(np.empty((n,)))
        for key in data['query_type'].keys():
            i = int(key)
            query_types[i] = data['query_type'][key]
        return query_types
    
    def _get_passages(self, data):
        """
        Gets the list of lists of passages from the loaded data.
        """
        n = len(data['passages'].keys())
        passages = list(np.empty((n,)))
        for key in data['passages'].keys():
            i = int(key)
            passages[i] = [passage['passage_text'] for passage in data['passages'][key]]
        return passages
    
    def _get_answers(self, data):
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
    
    def _get_well_formed_answers(self, data):
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
    
    def _get_combined_answers(self, answers, well_formed_answers):
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
        

def get_num_empty_answers(data_interface):
    """
    Gets the number of empty answers in the data interface.
    """
    count = 0
    for i in range(len(data_interface)):
        if len(data_interface[i]['answers']) == 0:
            count += 1
    return count


def element_to_prompt(element):
    """
    Converts an element of the data interface to a prompt for a language model.
    """
    # Providing the model the context passages.
    output = 'With the following passages:\n'
    for passage in element['passages']:
        output += passage + '\n\n'
    # Providing the model the query.
    query = element['query']
    output += f'Please answer this query: {query}\n'
    # Giving message about no answer.
    no_answer_phrase = 'No Answer Present.'
    output += f'If it is not possible to answer the query using the given prompts, please state: {no_answer_phrase}\n'
    return output


def write_n_prompts(data_interface, n, output):
    """
    Samples n prompts from n random elements in the data interface, to be given to a language model.
    Writes the output as a JSON file.
    """
    elements = data_interface.sample(n)
    prompts = [element_to_prompt(element) for element in elements]
    with open(output, 'w+') as f:
        json.dump(prompts, f, indent=4)


def plot_character_frequencies(data_interface):
    """
    Gets the frequency of each character within the answers for a data interface.
    """
    # Getting the number of each character and storing in a dict.
    char_counts = {}
    num_chars = 0
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            for character in answer.replace(' ', ''):
                # Grouping all cases together, and filtering out non-alphabetic characters.
                character = character.lower()
                if character in 'abcdefghijklmnopqrstuvwxyz':
                    num_chars += 1
                    if character not in char_counts:
                        char_counts[character] = 1
                    else:
                        char_counts[character] += 1
    # Sorting the char_counts by value, converting count to frequency, and storing it in paired x and y lists.
    sorted_items = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    x, y = zip(*sorted_items)
    y = [count / num_chars for count in y]
    # Sorting the items in the x and y 
    plt.bar(x, y)
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.show()
    top_10 = dict(sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    return top_10.keys()


def plot_word_length_kde(data_interface):
    """
    Pots the KDE for word length.
    """
    # Getting a list of all of the word lengths.
    word_lengths = []
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            for word in answer.split(' '):
                word_lengths.append(len(word))
    sns.kdeplot(data=word_lengths)
    plt.xlabel('Overall Word Lengths')
    plt.ylabel('Density')
    plt.xlim(0, 30)
    plt.show()
    return sum(word_lengths) / len(word_lengths)


def plot_sentence_length_kde(data_interface):
    """
    Pots the KDE for sentance length.
    """
    # Getting a list of all of the sentance lengths.
    sentance_lengths = []
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            for sentance in answer.split('.'):
                sentance_lengths.append(len(sentance))
    sns.kdeplot(data=sentance_lengths)
    plt.xlabel('Overall Sentance Length')
    plt.ylabel('Density')
    plt.xlim(0, 300)
    plt.show()
    return sum(sentance_lengths) / len(sentance_lengths)


def plot_num_letters_kde(data_interface):
    """
    Pots the KDE for response length.
    """
    # Getting a list of all of the sentance lengths.
    response_lengths = []
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            response_lengths.append(len(answer))
    sns.kdeplot(data=response_lengths)
    plt.xlabel('Per-Response Letter Count')
    plt.ylabel('Density')
    plt.xlim(0, 500)
    plt.show()
    return sum(response_lengths) / len(response_lengths)


def plot_num_words_kde(data_interface):
    """
    Pots the KDE for number of words per response.
    """
    # Getting a list of all of the word counts.
    word_counts = []
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            word_counts.append(len(answer.split(' ')))
    sns.kdeplot(data=word_counts)
    plt.xlabel('Per-Response Word Count')
    plt.ylabel('Density')
    plt.xlim(0, 100)
    plt.show()
    return sum(word_counts) / len(word_counts)


def plot_num_sentances_kde(data_interface):
    """
    Pots the KDE for number of sentances per response.
    """
    # Getting a list of all of the sentance counts.
    sentance_counts = []
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            sentance_counts.append(len(answer.split('.')))
    sns.kdeplot(data=sentance_counts)
    plt.xlabel('Per-Response Sentance Count')
    plt.ylabel('Density')
    plt.xlim(0, 7)
    plt.show()
    return sum(sentance_counts) / len(sentance_counts)


def reduce_answer_corpus(corpus, p):
    """
    Randomly samples a corpus down to a smaller one, with proportion p.
    """
    newsize = round(len(corpus) * p)
    return random.sample(corpus, newsize)


def get_answer_corpus(data_interface):
    """
    Gets the corpus for the answers in the data_interface
    """
    corpus = []
    for element in data_interface.list():
        answers = element['answers']
        for answer in answers:
            corpus.append(answer)
    return corpus


def apply_tf_idf(data_interface, p):
    """
    Applies tf-idf method to the data interface over the answers.
    """
    # Getting the answers as a corpus
    corpus = get_answer_corpus(data_interface)
    print(f'Original corpus length: {len(corpus)}')
    corpus = reduce_answer_corpus(corpus, p)
    print(f'Reduced corpus length: {len(corpus)}')
    # Performing tf-idf
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    # Re-formatting the output
    feature_names = vectorizer.get_feature_names_out()
    print(f'Number of distinct words: {len(feature_names)}')
    denselist = x.todense().tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    print(f'Datframe shape: {df.shape}')
    # Finding the words with the highest values
    sums = df.sum(axis=0)
    data = []
    for col, s in zip(sums.index, sums):
        data.append((col, s))
    ranking = pd.DataFrame(data, columns=['word', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)
    top_20_words = ranking.head(20).loc[:, 'word']
    bottom_20_words = ranking.tail(20).loc[:, 'word']
    print(f'Top 20 words from TF-IDF: {top_20_words.tolist()}')
    print(f'Bottom 20 words from TF-IDF: {bottom_20_words.tolist()}')


def main():
    data_file = '../data/dev_v2.1.json'
    data_interface = DataInterface(data_file)
    empty_answers = get_num_empty_answers(data_interface)
    total_answers = len(data_interface)
    print(f'Empty answers: {empty_answers} empty answers out of {total_answers} total answers.')
    print(f'Top 10 most frequent characters: {plot_character_frequencies(data_interface)}')
    print(f'Average word length overall: {plot_word_length_kde(data_interface)}')
    print(f'Average sentance length overall: {plot_sentence_length_kde(data_interface)}')
    print(f'Letter count per-response: {plot_num_letters_kde(data_interface)}')
    print(f'Word count per-response: {plot_num_words_kde(data_interface)}')
    print(f'Sentance count per-response: {plot_num_sentances_kde(data_interface)}')
    apply_tf_idf(data_interface, 0.2)


if __name__ == '__main__':
    main()
