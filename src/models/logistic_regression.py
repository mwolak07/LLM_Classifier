from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from numpy import ndarray
from typing import Tuple
from tqdm import tqdm
import numpy as np
from src.dataset import load_fasttext_data, pad_fasttext_data
from src.util import cd_to_executing_file, get_batches


def get_data() -> \
        Tuple[ndarray[ndarray[ndarray[np.float16]]], ndarray[ndarray[ndarray[np.float16]]], ndarray[int], ndarray[int]]:
    """
    Gets the data from the train and test databases, and applies the tokenizer for the model.

    Returns:
        x_train, x_test, y_train, y_test.
    """
    train_file_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_train_fasttext.npy'
    test_file_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_test_fasttext.npy'
    train_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3'
    test_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3'
    # Loading the vectorized data from disk, in 16 bit floats.
    x_train, x_test, y_train, y_test = load_fasttext_data(train_file_path, test_file_path, train_db_path, test_db_path)
    # Zero-padding the features.
    x_train, x_test = pad_fasttext_data(x_train, x_test)
    # Flattening the word vectors into one long sentence vector, as we need 2D data.
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    return x_train, x_test, y_train, y_test


def train(model: LogisticRegression, x: ndarray[ndarray[ndarray[np.float16]]], y: ndarray[int], batch_size: int) -> \
        LogisticRegression:
    """
    Trains the given GaussianNB model on the given training features and labels.

    Args:
        model: The sklearn model we are using to train.
        x: The set of training features.
        y: The set of training labels.
        batch_size: The size of each of the training batches.

    Returns:
        The model after training, with the learned weights.
    """
    x_batches = get_batches(x, batch_size)
    y_batches = get_batches(y, batch_size)
    for i in tqdm(range(len(x_batches))):
        x_batch = x_batches[i]
        y_batch = y_batches[i]
        model = model.partial_fit(x_batch, y_batch)
    return model


def test(model: LogisticRegression, x: ndarray[ndarray[ndarray[np.float16]]], y: ndarray[int]) -> None:
    """
    Tests the trained model with the given testing features and labels.

    Args:
        model: The sklearn model with learned weights (post-training) we are testing.
        x: The set of testing features.
        y: The set of testing labels.
    """
    predictions = model.predict(x)
    auc = metrics.roc_auc_score(y_true=y, y_score=predictions)
    precision = metrics.precision_score(y_true=y, y_pred=predictions)
    recall = metrics.recall_score(y_true=y, y_pred=predictions)
    f1 = metrics.f1_score(y_true=y, y_pred=predictions)
    accuracy = metrics.accuracy_score(y_true=y, y_pred=predictions)
    print(f'auc: {auc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    print(f'accuracy: {accuracy}')


def logistic_regression(max_iter: int, batch_size: int) -> None:
    """
    Trains and tests a gaussian naive bayes model.

    Args:
        max_iter: The number of iterations to allow the Logistic Regression to converge.
        batch_size: The size of each of the training batches.
    """
    cd_to_executing_file(__file__)
    model = LogisticRegression(penalty='l2', max_iter=max_iter)
    print(f'Loading data...')
    x_train, x_test, y_train, y_test = get_data()
    print(f'Training model...')
    train(model, x_train, y_train, batch_size)
    print(f'Testing model...')
    test(model, x_test, y_test)


if __name__ == '__main__':
    logistic_regression(max_iter=250, batch_size=256)
