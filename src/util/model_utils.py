from keras.callbacks import Callback
from typing import Any, Dict, List
import numpy as np


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
    Splits the given array into batches of size batch_size.

    Args:
        array: The array to be split into batches.
        batch_size: The size of each batch we want to be generating.

    Returns:
        A list of lists of the datatype of the input array, representing the list of batches.
    """
    batch_indexes = np.arange(batch_size, len(array), batch_size)
    batches = np.array_split(np.array(array), batch_indexes)
    return [batch.tolist() for batch in batches]
