from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from src import MSMarcoDataset
import seaborn as sns
import pandas as pd
import random
import json
import sys


"""
This section is for summarizing basic info about each of the JSON files for the MS MARCO dataset.
"""


def crop_line_to_str(line, line_n, tab_n):
    """
    Crops the line to the given line length, adding new lines where necessary, along with the correct indent.
    """
    line_len = line_n - tab_n
    tab = ' ' * tab_n
    # Account for empty string.
    if len(line) == 0:
        return tab + '\n'
    # Crop long lines.
    new_lines = []
    i = 0
    j = min(line_len, len(line))
    while i != j:
        new_lines.append(line[i: j])
        i = j
        j = min(j + line_len, len(line))
    # Converting the list of new lines to a '\n' separated string.
    output = ''
    for new_line in new_lines:
        output += tab + new_line + '\n'
    return output


def crop_labeled_line_to_str(label, line, line_n, tab_n):
    """
    Crops the line to the given line length, adding new lines where necessary, along with the correct indent and a
    label at the start.
    """
    line_len = line_n - tab_n - len(label)
    tab = ' ' * (tab_n + len(label))
    # Account for empty string.
    if len(line) == 0:
        return ' ' * tab_n + label + '\n'
    # Crop long lines.
    new_lines = []
    i = 0
    j = min(line_len, len(line))
    while i != j:
        new_lines.append(line[i: j])
        i = j
        j = min(j + line_len, len(line))
    # Converting the list of new lines to a '\n' separated string.
    output = ' ' * tab_n + label + new_lines[0] + '\n'
    if len(new_lines) > 1:
        for new_line in new_lines[1:]:
            output += tab + new_line + '\n'
    return output


def iterable_to_str(iterable):
    """
    Converts an iterable to a nice string.
    """
    output = ''
    for item in iterable:
        output += ' ' + item
    return output[1:]


def load_dataset(file):
    """
    Loads the data from the set in the given file.
    """
    with open(file, 'r') as f:
        return json.load(f)


def dataset_is_full(dataset):
    """
    Determines if this dataset has all of the possible information.
    """
    return 'answers' in dataset.keys() and 'wellFormedAnswers' in dataset.keys()


def get_data_categories(dataset):
    """
    Gets the data categories in the given dataset.
    """
    return dataset.keys()


def get_num_elements(dataset):
    """
    Gets the number of elements in this dataset.
    """
    num_passages = len(dataset['passages'].keys())
    num_queries = len(dataset['query'].keys())
    num_query_ids = len(dataset['query_id'].keys())
    num_query_types = len(dataset['query_type'].keys())
    if not num_passages == num_queries == num_query_ids == num_query_types:
        print('WARNING: element counts not identical for each data category!')
    return num_passages


def get_query_types(dataset):
    """
    Gets all of the query types for a given dataset, and packages them in a dict with the first index key for that query
     type.
    """
    query_types_dataset = dataset['query_type']
    # Getting all of the possible query types in the dataset.
    query_types = []
    for query_type in query_types_dataset.values():
        if query_type not in query_types:
            query_types.append(query_type)
    # Getting the index key of the first occurrence of query type.
    result = {}
    for index_key in query_types_dataset.keys():
        for query_type in query_types:
            if query_type not in result and query_types_dataset[index_key] == query_type:
                result[query_type] = index_key
    return result


def get_limited_dataset_sample(dataset):
    """
    Gets a sample of the limited dataset. This means the first passage, url, and query for each query type.
    """
    query_types = get_query_types(dataset)
    result = {}
    for query_type in query_types.keys():
        index_key = query_types[query_type]
        passage = dataset['passages'][index_key][0]['passage_text']
        url = dataset['passages'][index_key][0]['url']
        query = dataset['query'][index_key]
        result[query_type] = {'passage': passage, 'url': url, 'query': query}
    return result


def get_full_dataset_sample(dataset):
    """
    Gets a sample of the full dataset. This means the selected passages, url, query, answer, and well-formed answer
    for each query
    type.
    """
    query_types = get_query_types(dataset)
    result = {}
    for query_type in query_types.keys():
        index_key = query_types[query_type]
        passages = []
        urls = []
        for passage in dataset['passages'][index_key]:
            if passage['is_selected'] == 1:
                passages.append(passage['passage_text'])
                urls.append(passage['url'])
        answers = []
        for answer in dataset['answers'][index_key]:
            if answer != 'No Answer Present.':
                answers.append(answer)
        wellFormedAnswers = []
        if dataset['wellFormedAnswers'][index_key] != '[]':
            for wellFormedAnswer in dataset['wellFormedAnswers'][index_key]:
                wellFormedAnswers.append(wellFormedAnswer)
        query = dataset['query'][index_key]
        result[query_type] = {'passages': passages, 'urls': urls, 'query': query, 'answers': answers,
                              'wellFormedAnswers': wellFormedAnswers}
    return result


def get_limited_dataset_sample_string(dataset, tab_n, line_n):
    """
    Gets a string representing the dataset sample of the given limited dataset.
    """
    tab = ' ' * tab_n
    # Building the output string.
    output = ''
    dataset_sample = get_limited_dataset_sample(dataset)
    for query_type in dataset_sample.keys():
        # Adding query type.
        output += query_type + '\n'
        # Adding the query section, indented once.
        output += tab + 'Query: ' + dataset_sample[query_type]['query'] + '\n'
        # Adding the passage section, indented once.
        output += tab + 'Passage:' + '\n'
        # Adding the passage, indented twice.
        output += crop_line_to_str(dataset_sample[query_type]['passage'], line_n, tab_n * 2)
        # Adding the url section, indented twice.
        output += crop_labeled_line_to_str('URL: ', dataset_sample[query_type]['url'], line_n, tab_n * 2)
        # Adding a newline for readability.
        output += '\n'
    return output


def get_full_dataset_sample_string(dataset, tab_n, line_n):
    """
    Gets a string representing the dataset sample of the given full dataset.
    """
    tab = ' ' * tab_n
    # Building the output string.
    output = ''
    dataset_sample = get_full_dataset_sample(dataset)
    for query_type in dataset_sample.keys():
        # Adding query type.
        output += query_type + '\n'
        # Adding the query section, indented once.
        output += tab + 'Query: ' + dataset_sample[query_type]['query'] + '\n'
        # Adding the answer section section, indented once.
        answers = iterable_to_str(dataset_sample[query_type]['answers'])
        output += crop_labeled_line_to_str('Answers: ', answers, line_n, tab_n)
        # Adding the well formed answer section section, indented once.
        wellFormedAnswers = iterable_to_str(dataset_sample[query_type]['wellFormedAnswers'])
        output += crop_labeled_line_to_str('Well formed answers: ', wellFormedAnswers, line_n, tab_n)
        # Adding the passages section, indented once.
        output += tab + 'Passages:' + '\n'
        for i in range(len(dataset_sample[query_type]['passages'])):
            # Adding the passage, indented twice.
            output += crop_line_to_str(dataset_sample[query_type]['passages'][i], line_n, tab_n * 2)
            # Adding the url, indented twice.
            output += crop_labeled_line_to_str('URL: ', dataset_sample[query_type]['urls'][i], line_n, tab_n * 2)
            # Adding newline for readability.
            output += '\n'
        # Adding a newline for readability.
        output += '\n'
    return output


def write_dataset_info(input_file, output_file, tab_n, line_n):
    """
    Writes the info for the limited dataset at input_file to the output_file.
    """
    # Reading the info from the dataset.
    dataset = load_dataset(input_file)
    print('Dataset loaded!')
    num_elements = get_num_elements(dataset)
    data_categories = iterable_to_str(get_data_categories(dataset))
    query_types = iterable_to_str(get_query_types(dataset))
    if dataset_is_full(dataset):
        dataset_sample_string = get_full_dataset_sample_string(dataset, tab_n, line_n)
    else:
        dataset_sample_string = get_limited_dataset_sample_string(dataset, tab_n, line_n)
    # Writing the info to our output file.
    with open(output_file, 'w+') as f:
        f.write(f'Num elements: {num_elements}\n')
        f.write(crop_labeled_line_to_str('Data categories: ', data_categories, line_n, 0))
        f.write(crop_labeled_line_to_str('Query types: ', query_types, line_n, 0))
        f.write('Sample:\n')
        f.write(dataset_sample_string)


def get_dataset_info():
    if len(sys.argv) == 1:
        input_file_1 = '../data/eval_v2.1_public.json'
        input_file_2 = '../data/dev_v2.1.json'
        input_file_3 = '../data/train_v2.1.json'
        output_file_1 = '../data/eval_v2.1_public.info.txt'
        output_file_2 = '../data/dev_v2.1.json.info.txt'
        output_file_3 = '../data/train_v2.1.info.txt'
        write_dataset_info(input_file_1, output_file_1, 4, 100)
        write_dataset_info(input_file_2, output_file_2, 4, 100)
        write_dataset_info(input_file_3, output_file_3, 4, 100)
    elif len(sys.argv) < 3:
        print('Please input input file name followed by output file name when calling this script.')
    else:
        write_dataset_info(sys.argv[1], sys.argv[2], 4, 100)


"""
This section is for exploratory analysis of the MS MARCO dataset, with various techniques.
"""


def get_num_empty_answers(dataset):
    """
    Gets the number of empty answers in the data interface.
    """
    count = 0
    for i in range(len(dataset)):
        if len(dataset[i]['answers']) == 0:
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


def write_n_prompts(dataset, n, output):
    """
    Samples n prompts from n random elements in the data interface, to be given to a language model.
    Writes the output as a JSON file.
    """
    prompts = [dataset.prompt[i] for i in range(n)]
    with open(output, 'w+') as f:
        json.dump(prompts, f, indent=4)


def plot_character_frequencies(dataset):
    """
    Gets the frequency of each character within the answers for a data interface.
    """
    # Getting the number of each character and storing in a dict.
    char_counts = {}
    num_chars = 0
    for element in dataset.list():
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
    sorted_items = sorted(char_counts.items(), key=lambda item: item[1], reverse=True)
    x, y = zip(*sorted_items)
    y = [count / num_chars for count in y]
    # Sorting the items in the x and y 
    plt.bar(x, y)
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.show()
    top_10 = dict(sorted(char_counts.items(), key=lambda item: item[1], reverse=True)[:10])
    return top_10.keys()


def plot_word_length_kde(dataset):
    """
    Pots the KDE for word length.
    """
    # Getting a list of all of the word lengths.
    word_lengths = []
    for element in dataset.list():
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


def plot_sentence_length_kde(dataset):
    """
    Pots the KDE for sentence length.
    """
    # Getting a list of all of the sentence lengths.
    sentence_lengths = []
    for element in dataset.list():
        answers = element['answers']
        for answer in answers:
            for sentence in answer.split('.'):
                sentence_lengths.append(len(sentence))
    sns.kdeplot(data=sentence_lengths)
    plt.xlabel('Overall sentence Length')
    plt.ylabel('Density')
    plt.xlim(0, 300)
    plt.show()
    return sum(sentence_lengths) / len(sentence_lengths)


def plot_num_letters_kde(dataset):
    """
    Pots the KDE for response length.
    """
    # Getting a list of all of the sentence lengths.
    response_lengths = []
    for element in dataset.list():
        answers = element['answers']
        for answer in answers:
            response_lengths.append(len(answer))
    sns.kdeplot(data=response_lengths)
    plt.xlabel('Per-Response Letter Count')
    plt.ylabel('Density')
    plt.xlim(0, 500)
    plt.show()
    return sum(response_lengths) / len(response_lengths)


def plot_num_words_kde(dataset):
    """
    Pots the KDE for number of words per response.
    """
    # Getting a list of all of the word counts.
    word_counts = []
    for element in dataset.list():
        answers = element['answers']
        for answer in answers:
            word_counts.append(len(answer.split(' ')))
    sns.kdeplot(data=word_counts)
    plt.xlabel('Per-Response Word Count')
    plt.ylabel('Density')
    plt.xlim(0, 100)
    plt.show()
    return sum(word_counts) / len(word_counts)


def plot_num_sentences_kde(dataset):
    """
    Pots the KDE for number of sentences per response.
    """
    # Getting a list of all of the sentence counts.
    sentence_counts = []
    for element in dataset.list():
        answers = element['answers']
        for answer in answers:
            sentence_counts.append(len(answer.split('.')))
    sns.kdeplot(data=sentence_counts)
    plt.xlabel('Per-Response sentence Count')
    plt.ylabel('Density')
    plt.xlim(0, 7)
    plt.show()
    return sum(sentence_counts) / len(sentence_counts)


def reduce_answer_corpus(corpus, p):
    """
    Randomly samples a corpus down to a smaller one, with proportion p.
    """
    newsize = round(len(corpus) * p)
    return random.sample(corpus, newsize)


def get_answer_corpus(dataset):
    """
    Gets the corpus for the answers in the dataset
    """
    corpus = []
    for element in dataset.list():
        answers = element['answers']
        for answer in answers:
            corpus.append(answer)
    return corpus


def apply_tf_idf(dataset, p):
    """
    Applies tf-idf method to the data interface over the answers.
    """
    # Getting the answers as a corpus
    corpus = get_answer_corpus(dataset)
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
    print(f'Dataframe shape: {df.shape}')
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


def exploratory_analysis():
    data_file = '../data/dev_v2.1.json'
    dataset = MSMarcoDataset(data_file)
    empty_answers = get_num_empty_answers(dataset)
    total_answers = len(dataset)
    print(f'Empty answers: {empty_answers} empty answers out of {total_answers} total answers.')
    print(f'Top 10 most frequent characters: {plot_character_frequencies(dataset)}')
    print(f'Average word length overall: {plot_word_length_kde(dataset)}')
    print(f'Average sentence length overall: {plot_sentence_length_kde(dataset)}')
    print(f'Letter count per-response: {plot_num_letters_kde(dataset)}')
    print(f'Word count per-response: {plot_num_words_kde(dataset)}')
    print(f'sentence count per-response: {plot_num_sentences_kde(dataset)}')
    apply_tf_idf(dataset, 0.2)


if __name__ == '__main__':
    get_dataset_info()
    exploratory_analysis()
