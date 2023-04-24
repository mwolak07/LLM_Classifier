from keras.callbacks import Callback
from typing import Dict, Any


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
        epoch_filepath = f'{self.filepath[:extension_index]}_epoch_{epoch}.{extension}'
        self.model.save_weights(filepath=epoch_filepath, save_format=self.save_format)
        if self.verbose:
            print(f'Saving weights for epoch {epoch} to {epoch_filepath}')
