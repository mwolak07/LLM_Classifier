from typing import Any, Dict, List, Union
from keras.callbacks import Callback
from numpy import ndarray
import numpy as np


# Storing the type definition for a Feature, to make things simpler.
Feature = Union[str, ndarray[ndarray[np.float32]]]


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


def get_batches(array: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Splits the given array into batches of size batch_size..

    Args:
        array: The array to be split into batches.
        batch_size: The size of each batch of questions we want to be generating.

    Returns:
        A list of lists of the datatype of the input array, representing the list of batches.
    """
    batch_indexes = np.arange(batch_size, len(array), batch_size)
    batches = np.array_split(np.array(array), batch_indexes)
    return [batch.tolist() for batch in batches]


def fasttext_pad(text: ndarray[ndarray[np.float32]], max_words: int) -> ndarray[ndarray[np.float32]]:
    """
    Zero-pads the given array of words (fasttext vectors), so that the number of words in the array is the same as the
    maximum. This is used to ensure when the text vectors for all of the samples are stacked, the shape is uniform
    (not ragged).

    Args:
        text: The block of text, in vectorized form, to be padded with zeros to max_words.
        max_words: The maximum amount of words in a sentence, the number we are padding to.

    Returns:
        The argument text padded with zeros to have max_words word vectors.
    """
    word_length = len(text[0])
    zeros = np.zeros((max_words - len(text), word_length), dtype=np.float32)
    return np.concatenate((text, zeros))
