from gensim.models.fasttext import load_facebook_model
from typing import Tuple, Any, Dict, List
from keras.callbacks import Callback
from numpy import ndarray
import pandas as pd
import numpy as np
from src.util import cd_to_executing_file
from src.dataset import LLMClassifierDataset


class SaveWeightsCallback(Callback):
    """
    Custom callback to save the model weights to a different file for each epoch.

    Attributes:
        filepath: The base filepath to use for saving the weights, that _epoch will be appended to.
        save_format: The format to save the weights to. Can be: 'h5', 'tf'
        verbose: Should we print something out at the end of each epoch?
    """
    filepath: str
    save_format: str
    verbose: bool

    def __init__(self, filepath: str, save_format: str, verbose: bool = False):
        super().__init__()
        if '.h5' not in filepath and '.keras' not in filepath:
            raise ValueError('File path must end with ".h5" or ".keras"!')
        self.filepath = filepath
        self.save_format = save_format
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        extension = '.' + self.filepath.split('.')[-1]
        extension_index = self.filepath.index(extension)
        epoch_filepath = f'{self.filepath[:extension_index]}_epoch_{epoch + 1}{extension}'
        self.model.save_weights(filepath=epoch_filepath, save_format=self.save_format)
        if self.verbose:
            print(f'Saving weights for epoch {epoch} to {epoch_filepath}')


def load_data(train_db_path: str, test_db_path: str) -> \
        Tuple[ndarray[str], ndarray[str], ndarray[int], ndarray[int]]:
    """
    Loads the data from the LLMClassifierDataset and separates the data into features and labels.

    Args:
        train_db_path: The path the the database containing the training data.
        test_db_path: The path to the database containing the testing data.

    Returns:
        x_train, x_test, y_train, y_test.
    """
    # Loading in and processing the data.
    cd_to_executing_file(__file__)
    train_dataset = LLMClassifierDataset(train_db_path)
    test_dataset = LLMClassifierDataset(test_db_path)
    train_dataset_items = train_dataset.tolist()
    test_dataset_items = test_dataset.tolist()
    # Separating our the features and labels.
    x_train = [item[0] for item in train_dataset_items]
    x_test = [item[0] for item in test_dataset_items]
    y_train = [item[1] for item in train_dataset_items]
    y_test = [item[1] for item in test_dataset_items]
    # Converting to numpy arrays.
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


def load_fasttext_data(train_file_path: str, test_file_path: str) -> \
        Tuple[List[List[ndarray[float]]], List[List[ndarray[float]]], ndarray[int], ndarray[int]]:
    """
    Loads the fasttext vectorized data from disk, where it was saved previously. This will be a ragged list, you can use
    pad_fasttext_vectors to pad x_train and x_text

    Args:
        train_file_path: The file path to the CSV file storing the vectorized training data.
        test_file_path: The file path to the CSV file storing the vectorized testing data.

    Returns:
        x_train, x_test, y_train, y_test.
    """
    # Loading the data from the CSVs.
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    # Unpacking the data.
    return train_df['x_train'], test_df['x_test'], train_df['y_train'], test_df['y_test']


def pad_fasttext_data(x_train: List[List[ndarray[float]]], x_test: List[List[ndarray[float]]]) \
        -> Tuple[ndarray[ndarray[ndarray[float]]], ndarray[ndarray[ndarray[float]]]]:
    """
    Zero-pads the given list of fasttext vectors, so that the number of vectors in the list is the same as the maximum.
    This is used to ensure when the text vectors for all of the samples are stacked, the shape is uniform (not ragged).
    Each vector in the list has the same length, so we add n zero vectors of this length to the end of the vector_list.

    Args:
        x_train: The list of lists of vectorized words representing blocks of text in the training set.
        x_test: The list of lists of vectorized words representing blocks of text in the testing set.

    Returns:
        x_train, x_test padded with zeros to all have the same number of word vectors.

    Raises:
        ValueError: If the size of a word vector is different in the two sets.
    """
    # Checking word vector lengths are consistent.
    if len(x_train[0][0]) != len(x_test[0][0]):
        raise ValueError(f'Word vector length mismatch between x_train and x_test! '
                         f'{len(x_train[0][0])} != {len(x_test[0][0])}!')
    # Getting the relevant lengths.
    word_vector_length = len(x_train[0][0])
    train_max_text_len = max([len(text) for text in x_train])
    test_max_text_len = max([len(text) for text in x_test])
    max_text_len = max(train_max_text_len, test_max_text_len)
    # Padding each text element, according to what is necessary.
    # Training set.
    x_train_output = np.empty((len(x_train), max_text_len, word_vector_length))
    for i in range(len(x_train)):
        text = x_train[i]
        zeros = np.zeros((max_text_len - len(text), word_vector_length), dtype=int)
        x_train_output[i] = np.concatenate((text, zeros))
    x_test_output = np.empty((len(x_test), max_text_len, word_vector_length))
    # Testing set.
    for i in range(len(x_test)):
        text = x_test[i]
        zeros = np.zeros((max_text_len - len(text), word_vector_length), dtype=int)
        x_test_output[i] = np.concatenate((text, zeros))
    # Returning the result
    return x_train_output, x_test_output


def write_fasttext_data(train_db_path: str, test_db_path: str) -> None:
    """
    Vectorizes the data from the LLMClassifierDataset using fasttext, and saves the result to disk, as CSVs of the
    training and testing datasets. Writes these CSVs to the same directory as the source databases.

    Args:
        train_db_path: The path the the database containing the training data.
        test_db_path: The path to the database containing the testing data.
    """
    # Determining the paths to save the CSV data to.
    extension = '.sqlite3'
    train_db_file = train_db_path.split('/')[-1].replace(extension, '')
    test_db_file = test_db_path.split('/')[-1].replace(extension, '')
    train_file_path = train_db_path.replace(train_db_file, '') + '_fasttext.csv'
    test_file_path = test_db_path.replace(test_db_file, '') + '_fasttext.csv'
    # Loading the datasets from the database.
    print(f'Loading the data...')
    x_train, x_test, y_train, y_test = load_data(train_db_path, test_db_path)
    # Vectorizing the feature sets.
    print(f'Vectorizing x_train...')
    x_train = fasttext_vectorize(x_train)
    print(f'Vectorizing x_test...')
    x_test = fasttext_vectorize(x_test)
    # Putting the fasttext data into a pandas CSV.
    print(f'Writing the data...')
    train_df = pd.DataFrame({'x_train': x_train, 'y_train': y_train})
    test_df = pd.DataFrame({'x_test': x_test, 'y_test': y_test})
    # Writing the CSV files.
    train_df.to_csv(train_file_path)
    test_df.to_csv(test_file_path)


def fasttext_vectorize(text_list: ndarray[str]) -> ndarray[float]:
    """
    Applies the pre-trained fasttext vectorizer as efficiently as possible to the given list of strings.

    Args:
        text_list: The list of blocks of text to be transformed into lists of vectors using the fasttext vectorizer.

    Returns:
        The list of strings transformed into vector representations.
    """
    print(f'Loading model...')
    # Setting up the vectorizer.
    cd_to_executing_file(__file__)
    fasttext_path = '../../data/fasttext/wiki.en.bin'
    vectorizer = load_facebook_model(fasttext_path)
    print(f'Vectorizing words...')
    # Iterating through each word in each block of text.
    output = []
    for text in text_list:
        text_vectors = []
        for word in text.split(' '):
            text_vectors.append(vectorizer.wv[word])
        output.append(np.array(text_vectors))
    return output
