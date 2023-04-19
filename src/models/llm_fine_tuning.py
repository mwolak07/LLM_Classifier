from keras.metrics import Accuracy, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from typing import Tuple, Any
from numpy import ndarray
import numpy as np
from src.dataset import LLMClassifierDataset
from src.util import cd_to_executing_file


def load_data(model_name: str) -> Tuple[ndarray[float], ndarray[float], ndarray[int], ndarray[int]]:
    """
    Loads the data from the LLMClassifierDataset, applies the pre-trained tokenizer, and creates the train and test
    data.

    Args:
        model_name: The huggingface name of the model for the pre-trained tokenizer.

    Returns:
        x_train, x_test, y_train, y_test.
    """
    # Loading in the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Loading in and processing the data.
    cd_to_executing_file(__file__)
    db_path = '../../data/bloom_1_1B/test_short_prompts.sqlite3'
    dataset = LLMClassifierDataset(db_path)
    features = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    # Tokenizing the features.
    features = np.array(tokenizer(features, return_tensors='np', padding='longest'))
    # Converting the labels to categorical vectors.
    labels = np.array(to_categorical(labels))
    # Splitting the data into a training and testing set.
    return train_test_split(features, labels, train_size=0.9)


def load_model(model_name) -> Model:
    """
    Loads and compiles the pre-trained LLM model given.

    Returns:
        The loaded and compiled model.
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.compile(optimizer=Adam(3e-5),
                  metrics=[Accuracy(),
                           AUC(),
                           TruePositives(),
                           TrueNegatives(),
                           FalsePositives(),
                           FalseNegatives()])



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
    # Loading in the model fitting it to the data.
    model.fit(x, y, batch_size=32, validation_split=0.2, epochs=epochs,
              callbacks=[TensorBoard(log_dir=f'../logs/llm_fine_tuning/{model_file}')])
    model.save_weights(filepath=f'../model_weights/{model_file}', format='h5')


def test(x: ndarray[float], y: ndarray[int], model_name: str) -> Any:
    """
    Tests the model with the vectorized text features and classification labels.

    Args:
        x: The testing features for the model to predict on.
        y: The testing labels to compare with.
        model_name: The huggingface name of the model we are training.
    """
    model_file = model_name.split('/')[-1]
    model = load_model(model_name)
    model.load_weights(f'../model_weights/{model_file}')
    return model.evaluate(x, y, batch_size=32)


def fine_tune_model(model_name: str) -> None:
    """
    Fine-tunes an LLM model, including training and testing.

    Args:
        model_name: The huggingface name of the model we are training.
    """
    x_train, x_test, y_train, y_test = load_data(model_name)
    train(x_train, y_train, model_name=model_name, epochs=25)
    print(test(x_test, y_test, model_name=model_name))


def main() -> None:



if __name__ == '__main__':