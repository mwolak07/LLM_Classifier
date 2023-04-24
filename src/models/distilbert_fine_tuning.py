from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Model
from typing import Tuple
from sklearn import metrics
from numpy import ndarray
import numpy as np
from src.util import cd_to_executing_file, SaveWeightsCallback
from src.dataset import LLMClassifierDataset


def load_data(model_name: str, db_path: str) -> Tuple[ndarray[float], ndarray[float], ndarray[int], ndarray[int]]:
    """
    Loads the data from the LLMClassifierDataset, applies the pre-trained tokenizer, and creates the train and test
    data.

    Args:
        model_name: The huggingface name of the model for the pre-trained tokenizer.
        db_path: The path the the database containing the data.

    Returns:
        x_train, x_test, y_train, y_test.
    """
    # Loading in the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # Loading in and processing the data.
    cd_to_executing_file(__file__)
    dataset = LLMClassifierDataset(db_path)
    features = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    # Splitting the data into a training and testing set.
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.9)
    # Tokenizing the features.
    x_train = dict(tokenizer(x_train, return_tensors='np', padding=True))
    x_test = dict(tokenizer(x_test, return_tensors='np', padding=True))
    # Converting the labels to categorical vectors.
    y_train = np.array(y_train)
    y_test = np.array(y_test)
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
    model_file = model_name.split('/')[-1]
    model = load_model(model_name)
    # Defining our callbacks.
    callbacks = [
        TensorBoard(log_dir=f'../logs/llm_fine_tuning/{model_file}'),
        SaveWeightsCallback(filepath=f'../model_weights/{model_file}/weights.h5',
                            save_format='h5', verbose=True)
    ]
    # Loading in the model fitting it to the data.
    model.fit(x, y, batch_size=2,
              validation_split=0.25, epochs=epochs, callbacks=callbacks)
    model.save_weights(filepath=f'../model_weights/{model_file}/weights.h5', save_format='h5')


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
    model.load_weights(f'../model_weights/{model_file}.h5')
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
    db_path = '../../data/bloom_1_1B/test_short_prompts.sqlite3'
    print(f'Loading data...')
    x_train, x_test, y_train, y_test = load_data(model_name, db_path)
    print(f'Training model...')
    train(x_train, y_train, model_name=model_name, epochs=epochs)
    print(f'Testing model...')
    print(test(x_test, y_test, model_name=model_name))


def main() -> None:
    fine_tune_model(25)


if __name__ == '__main__':
    main()
