from keras.utils import Sequence, to_categorical
from keras.layers import LSTM, Dense, Masking
from keras.losses import BinaryCrossentropy
from keras.models import Model, Sequential
from keras.metrics import BinaryAccuracy
from keras.callbacks import TensorBoard
from multiprocessing import cpu_count
from keras.optimizers import Adam
from typing import List, Tuple
from sklearn import metrics
from numpy import ndarray
import numpy as np
import random
import math
import os
from src.util import Feature, SaveWeightsCallback, get_batches, fasttext_pad, cd_to_executing_file
from src.dataset import LLMClassifierDataset


class LLMClassifierDataLoader(Sequence):
    """
    Represents a custom batched data loader for the LLM Classifier set. Pads the data according to max_words.

    Attributes:
        db_path: The path to the database.
        fasttext: If true, we use the fasttext vectorized versions of the features, instead of the strings.
        load_to_memory: If true, we will load the contents of the database into memory. This will only be beneficial if
                        our dataset is smaller than our RAM.
        batch_size: The batch size we are using.
        shuffle: Should we shuffle the dataset at the end of each epoch?
        _indexes: The list of indexes in the dataset, in the randomized order we will be sampling them in.
        _index_batches: The batches of indexes from self.indexes of size batch_size.
    """
    db_path: str
    fasttext: bool
    load_to_memory: str
    batch_size: int
    max_words: int
    shuffle: bool
    _indexes: List[int]
    _index_batches: List[List[int]]

    def __init__(self, db_path: str, batch_size: int, max_words: int, training_ratio: float, use_validation: bool,
                 training_set: bool, shuffle: bool = True, fasttext: bool = False, load_to_memory: bool = False):
        """
        Initializes this data loader at the given db loc with the given batch size.

        Args:
            db_path: The location on disk of the database we are getting our data from.
            batch_size: The size of each mini-batch in the dataset.
            max_words: The maximum number of words in an answer, for padding.
            training_ratio: The ratio of samples training: validation.
            use_validation: True if we should split into training and validation sets.
            training_set: True if this is the training set, false if this is the validation set.
            shuffle: Should be shuffle our dataset at the end of each epoch?
            fasttext: If true, we use the fasttext vectorized versions of the features, instead of the strings.
            load_to_memory: If true, we will load the contents of the database into memory. This will only be beneficial
                            if our dataset is smaller than our RAM.
        """
        self.db_path = db_path
        self.fasttext = fasttext
        self.load_to_memory = load_to_memory
        dataset = LLMClassifierDataset(db_path, fasttext, load_to_memory)
        self.batch_size = batch_size
        self.max_words = max_words
        self.shuffle = shuffle
        self._indexes = list(range(len(dataset)))
        # Shuffling the indexes if needed.
        if shuffle:
            random.shuffle(self._indexes)
        # Splitting into test/validation, based on the indexes.
        if use_validation:
            split_point = int(round(training_ratio * len(self._indexes)))
            if training_set:
                self._indexes = self._indexes[:split_point]
            else:
                self._indexes = self._indexes[split_point:]
        # Splitting the indexes up into batches.
        self._index_batches = get_batches(self._indexes, batch_size)

    def __getitem__(self, index: int) -> Tuple[ndarray[Feature], ndarray[ndarray[int]]]:
        """
        Gets the batch at the given index.

        Args:
            index: The index in self._index_batches to get the batch for.

        Returns:
            Batch of features, batch of categorical labels.
        """
        dataset = LLMClassifierDataset(self.db_path, self.fasttext, self.load_to_memory)
        # Constructing our batch.
        batch = []
        for i in self._index_batches[index]:
            batch.append(dataset[i])
        # Splitting into features and labels.
        x_batch = np.array([fasttext_pad(element[0], self.max_words) for element in batch], dtype=np.float32)
        y_batch = np.array([element[1] for element in batch], dtype=np.float32)
        # Changing our y_batch to a to_categorical.
        y_batch = to_categorical(y_batch, num_classes=2)
        return x_batch, y_batch

    def __len__(self) -> int:
        """
        Gets the number of batches.

        Returns:
            The number of batches.
        """
        return len(self._index_batches)

    def on_epoch_end(self) -> None:
        """
        Resets the dataset at the end of the epoch, shuffling the indices and re-calculating the batches if needed.
        """
        if self.shuffle:
            random.shuffle(self._indexes)
            self._index_batches = get_batches(self._indexes, self.batch_size)


def get_dataloaders(batch_size: int, training_ratio: float) -> \
        Tuple[LLMClassifierDataLoader, LLMClassifierDataLoader, LLMClassifierDataLoader, int, int]:
    """
    Gets the data from the train and test databases, and applies the tokenizer for the model.

    Args:
        batch_size: The size of the batches to use in the dataset.
        training_ratio: The ratio of samples training: validation.

    Returns:
        train_dataloader, validation_dataloader, test_dataloader, max_words, word_length
    """
    train_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3'
    test_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3'
    train_dataset = LLMClassifierDataset(db_path=train_db_path, fasttext=True, load_to_memory=False)
    test_dataset = LLMClassifierDataset(db_path=test_db_path, fasttext=True, load_to_memory=False)
    max_words = max(train_dataset.max_words(), test_dataset.max_words())
    word_length = len(train_dataset[0][0][0])
    return LLMClassifierDataLoader(db_path=train_db_path, batch_size=batch_size, max_words=max_words,
                                   training_ratio=training_ratio, use_validation=True, training_set=True, shuffle=True,
                                   fasttext=True, load_to_memory=False), \
           LLMClassifierDataLoader(db_path=train_db_path, batch_size=batch_size, max_words=max_words,
                                   training_ratio=training_ratio, use_validation=True, training_set=False, shuffle=True,
                                   fasttext=True, load_to_memory=False), \
           LLMClassifierDataLoader(db_path=test_db_path, batch_size=batch_size, max_words=max_words,
                                   training_ratio=training_ratio, use_validation=False, training_set=True,
                                   shuffle=False, fasttext=True, load_to_memory=False), \
           max_words, word_length


def load_model(max_words: int, word_length: int) -> Model:
    """
    Loads and compiles the LSTM model.

    Args:
        max_words: The maximum number of words in an input sequence.
        word_length: The maximum length of a word in the input sequence.

    Returns:
        The loaded and compiled model.

    """
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(max_words, word_length), dtype=np.float32))
    model.add(LSTM(units=word_length, activation='tanh', return_sequences=False, dtype=np.float32))
    model.add(Dense(units=32, activation='relu', dtype=np.float32))
    model.add(Dense(units=2, activation='softmax', dtype=np.float32))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])
    model.summary()
    return model


def train(train_dataloader: LLMClassifierDataLoader, validation_dataloader: LLMClassifierDataset, model_name: str,
          max_words: int, word_length: int, epochs: int) -> None:
    """
    Trains the model with the given name with the vectorized text features and classification labels.

    Args:
        train_dataloader: The dataloader for the training set.
        validation_dataloader: The dataloader for the validation set.
        model_name: The name of the model.
        max_words: The maximum number of words in an input sequence.
        word_length: The maximum length of a word in the input sequence.
        epochs: The number of epochs to train for.
    """
    # Creating callbacks and making our directories if needed.
    make_callback_dirs(model_name)
    callbacks = [
        TensorBoard(log_dir=f'../logs/{model_name}'),
        SaveWeightsCallback(filepath=f'../model_weights/{model_name}/weights.h5',
                            save_format='h5', verbose=False)
    ]
    # Calculating the number of workers such that all will have a batch on the validation set, to a maximum of the
    # number of cores.
    workers = min(len(validation_dataloader), cpu_count())
    print(f'Loading data with {workers} cores')
    # Loading in the model fitting it to the data.
    model = load_model(max_words, word_length)
    model.fit(train_dataloader, validation_data=validation_dataloader, epochs=epochs, callbacks=callbacks,
              workers=workers, use_multiprocessing=True, max_queue_size=1)
    model.save_weights(filepath=f'../model_weights/{model_name}/weights.h5', save_format='h5')


def make_callback_dirs(model_name: str) -> None:
    """
    Creates the directories for the callbacks.

    Args:
        model_name: The name of the model we are training.
    """
    cd_to_executing_file(__file__)
    # Logs
    if not os.path.exists(f'../logs'):
        os.mkdir(f'../logs')
    if not os.path.exists(f'../logs/{model_name}'):
        os.mkdir(f'../logs/{model_name}')
    # Weights
    if not os.path.exists(f'../model_weights'):
        os.mkdir(f'../model_weights')
    if not os.path.exists(f'../model_weights/{model_name}'):
        os.mkdir(f'../model_weights/{model_name}')


def test(test_dataloader: LLMClassifierDataLoader, model_name: str, max_words: int, word_length: int) -> None:
    """
    Tests the model with the given name with the vectorized text features and classification labels.

    Args:
        test_dataloader: The dataloader for the testing set.
        model_name: The name of the model.
        max_words: The maximum number of words in an input sequence.
        word_length: The maximum length of a word in the input sequence.
    """
    model = load_model(max_words, word_length)
    model.load_weights(f'../model_weights/{model_name}/weights_epoch_10.h5')
    logits = model.predict(test_dataloader)
    predictions = np.argmax(logits, axis=1)
    y = np.array([np.argmax(element) for batch in test_dataloader for element in batch[1]], dtype=int)
    auc = metrics.roc_auc_score(y_true=y, y_score=predictions)
    precision = metrics.precision_score(y_true=y, y_pred=predictions)
    recall = metrics.recall_score(y_true=y, y_pred=predictions)
    f1 = metrics.f1_score(y_true=y, y_pred=predictions)
    evaluation = model.evaluate(test_dataloader)
    print(f'auc: {auc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    print(f'accuracy: {evaluation[1]}')
    print(f'loss: {evaluation[0]}')


def lstm(epochs: int, batch_size: int) -> None:
    """
    Trains and tests the LSTM model.

    Args:
        epochs: The number of epochs to train for.
        batch_size: The size of each of the training batches.
    """
    cd_to_executing_file(__file__)
    model_name = 'lstm'
    print(f'Loading data...')
    train_dataloader, validation_dataloader, test_dataloader, max_words, word_length = \
        get_dataloaders(batch_size=batch_size, training_ratio=0.75)
    print(f'Training model...')
    train(train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, model_name=model_name,
          max_words=max_words, word_length=word_length, epochs=epochs)
    print(f'Testing model...')
    test(test_dataloader=test_dataloader, model_name=model_name, max_words=max_words, word_length=word_length)


if __name__ == '__main__':
    lstm(25, 512)
