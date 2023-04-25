from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from keras.callbacks import TensorBoard
from typing import Tuple, Dict, Any
from keras.optimizers import Adam
from keras.models import Model
from sklearn import metrics
from numpy import ndarray
import numpy as np
import os
from src.util import cd_to_executing_file, SaveWeightsCallback, load_data


def get_data(model_name: str, train_db_path: str, test_db_path: str) -> \
        Tuple[Dict[Any], Dict[Any], ndarray[int], ndarray[int]]:
    """
    Gets the data from the train and test databases, and applies the tokenizer for the model.

    Args:
        model_name: The huggingface name of the model we are loading data for.
        train_db_path: The path to the training database.
        test_db_path: The path to the testing database.

    Returns:
        x_train, x_test, y_train, y_test.
    """
    # Loading in the data.
    x_train, x_test, y_train, y_test = load_data(train_db_path, test_db_path)
    # Loading in the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenizing the features.
    x_train = dict(tokenizer(x_train.tolist(), return_tensors='np', padding=True, truncation=True))
    x_test = dict(tokenizer(x_test.tolist(), return_tensors='np', padding=True, truncation=True))
    return x_train, x_test, y_train, y_test


def load_model(model_name) -> Model:
    """
    Loads and compiles the pre-trained LLM model given.

    Returns:
        The loaded and compiled model.
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    model.compile(optimizer=Adam(3e-5), metrics=['accuracy'])
    return model


def train(x: ndarray[float], y: ndarray[int], model_name: str, epochs: int) -> None:
    """
    Trains the given llm model with vectorized text features and classification labels.

    Args:
        x: The training features for the model to fit.
        y: The training labels for the model to fit.
        model_name: The huggingface name of the model we are training.
        epochs: The number of epochs to train for.
    """
    cd_to_executing_file(__file__)
    model_file = model_name.split('/')[-1]
    # Creating callbacks and making our directories if needed.
    make_callback_dirs(model_name)
    callbacks = [
        TensorBoard(log_dir=f'../logs/{model_file}'),
        SaveWeightsCallback(filepath=f'../model_weights/{model_file}/weights.h5',
                            save_format='h5', verbose=False)
    ]
    # Loading in the model fitting it to the data.
    model = load_model(model_name)
    model.fit(x, y, batch_size=8, validation_split=0.25, epochs=epochs, callbacks=callbacks)
    model.save_weights(filepath=f'../model_weights/{model_file}/weights.h5', save_format='h5')


def make_callback_dirs(model_name: str) -> None:
    """
    Creates the directories for the callbacks.

    Args:
        model_name: The huggingface name of the model we are training.
    """
    cd_to_executing_file(__file__)
    model_file = model_name.split('/')[-1]
    # Logs
    if not os.path.exists(f'../logs'):
        os.mkdir(f'../logs')
    if not os.path.exists(f'../logs/{model_file}'):
        os.mkdir(f'../logs/{model_file}')
    # Weights
    if not os.path.exists(f'../model_weights'):
        os.mkdir(f'../model_weights')
    if not os.path.exists(f'../model_weights/{model_file}'):
        os.mkdir(f'../model_weights/{model_file}')


def test(x: ndarray[float], y: ndarray[int], model_name: str) -> None:
    """
    Tests the model with the vectorized text features and classification labels.

    Args:
        x: The testing features for the model to predict on.
        y: The testing labels to compare with.
        model_name: The huggingface name of the model we are training.
    """
    model_file = model_name.split('/')[-1]
    model = load_model(model_name)
    model.load_weights(f'../model_weights/{model_file}/weights.h5')
    logits = model.predict(x)['logits']
    predictions = np.argmax(logits, axis=1)
    auc = metrics.roc_auc_score(y_true=y, y_score=predictions)
    precision = metrics.precision_score(y_true=y, y_pred=predictions)
    recall = metrics.recall_score(y_true=y, y_pred=predictions)
    f1 = metrics.f1_score(y_true=y, y_pred=predictions)
    evaluation = model.evaluate(x, y, batch_size=2)
    print(f'auc: {auc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    print(f'evaluation: {evaluation}')


def fine_tune_model(epochs: int) -> None:
    """
    Fine-tunes an LLM model, including training and testing.

    Args:
        epochs: The number of epochs to train for.
    """
    model_name = 'distilbert-base-cased'
    train_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_train.sqlite3'
    test_db_path = '../../data/bloom_1_1B/dev_v2.1_short_prompts_test.sqlite3'
    print(f'Loading data...')
    x_train, x_test, y_train, y_test = get_data(model_name, train_db_path=train_db_path, test_db_path=test_db_path)
    print(f'Training model...')
    train(x_train, y_train, model_name=model_name, epochs=epochs)
    print(f'Testing model...')
    print(test(x_test, y_test, model_name=model_name))


def main() -> None:
    fine_tune_model(25)


if __name__ == '__main__':
    main()
