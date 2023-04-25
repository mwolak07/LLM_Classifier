from keras.callbacks import Callback
from typing import Tuple, Any, Dict
from numpy import ndarray
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
        epoch_filepath = f'{self.filepath[:extension_index]}_epoch_{epoch + 1}.{extension}'
        self.model.save_weights(filepath=epoch_filepath, save_format=self.save_format)
        if self.verbose:
            print(f'Saving weights for epoch {epoch} to {epoch_filepath}')


def load_data(train_db_path: str, test_db_path: str) -> \
        Tuple[ndarray[float], ndarray[float], ndarray[int], ndarray[int]]:
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
