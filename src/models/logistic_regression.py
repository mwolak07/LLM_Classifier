from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from typing import Tuple
from tqdm import tqdm
import numpy as np
from src.util import cd_to_executing_file, get_batches, fasttext_pad
from src.dataset import LLMClassifierDataset


def get_datasets() -> Tuple[LLMClassifierDataset, LLMClassifierDataset]:
    """
    Gets the data from the train and test databases, and applies the tokenizer for the model.

    Returns:
        train_dataset, test_dataset
    """
    train_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3'
    test_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3'
    return LLMClassifierDataset(db_path=train_db_path, fasttext=True, load_to_memory=False), \
           LLMClassifierDataset(db_path=test_db_path, fasttext=True, load_to_memory=False)


def train(model: SGDClassifier, dataset: LLMClassifierDataset, max_words: int, batch_size: int) -> SGDClassifier:
    """
    Trains the given GaussianNB model on the given training dataset.

    Args:
        model: The sklearn model we are using to train.
        dataset: The dataset we will be using to train the model.
        max_words: The maximum number of words in an answer, for padding.
        batch_size: The size of each of the training batches.

    Returns:
        The model after training, with the learned weights.
    """
    # Constructing our batch indexes.
    indexes = list(range(len(dataset)))
    index_batches = get_batches(indexes, batch_size)
    for i in tqdm(range(len(index_batches))):
        # Constructing our batch.
        batch = []
        for j in index_batches[i]:
            batch.append(dataset[j])
        # Splitting into features and labels.
        x_batch = np.array([fasttext_pad(element[0], max_words) for element in batch])
        y_batch = np.array([element[1] for element in batch])
        # Flattening the features.
        x_batch = np.reshape(x_batch, (x_batch.shape[0], x_batch.shape[1] * x_batch.shape[2]))
        # Performing the training step.
        model = model.partial_fit(x_batch, y_batch, classes=np.array([0, 1]))
    return model


def test(model: SGDClassifier, dataset: LLMClassifierDataset, max_words: int, batch_size: int) -> None:
    """
    Tests the trained model with the given testing dataset.

    Args:
        model: The sklearn model with learned weights (post-training) we are testing.
        dataset: The dataset we will be using to test the model.
        max_words: The maximum number of words in an answer, for padding.
        batch_size: The size of each of the training batches.
    """
    y = [element[1] for element in dataset]
    # Getting the predictions in batches.
    predictions = np.array([])
    # Constructing our batch indexes.
    indexes = list(range(len(dataset)))
    index_batches = get_batches(indexes, batch_size)
    for i in tqdm(range(len(index_batches))):
        # Constructing our batch.
        batch = []
        for j in index_batches[i]:
            batch.append(dataset[j])
        # Splitting out the features.
        x_batch = np.array([fasttext_pad(element[0], max_words) for element in batch])
        # Flattening the features.
        x_batch = np.reshape(x_batch, (x_batch.shape[0], x_batch.shape[1] * x_batch.shape[2]))
        # Predicting on the features.
        predictions = np.concatenate((predictions, model.predict(x_batch)))
    # Computing the metrics.
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
    model = SGDClassifier(loss='log', penalty=None, max_iter=max_iter)
    print(f'Loading data...')
    train_dataset, test_dataset = get_datasets()
    print(f'Calculating max words...')
    max_words = max(train_dataset.max_words(), test_dataset.max_words())
    print(f'Training model...')
    train(model, train_dataset, max_words, batch_size)
    print(f'Testing model...')
    test(model, test_dataset, max_words, batch_size)


if __name__ == '__main__':
    logistic_regression(max_iter=250, batch_size=4096)
