from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from keras.metrics import Accuracy, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from src.dataset import LLMClassifierDataset
from src.util import cd_to_executing_file


# Loading in the tokenizer.
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

# Loading in and processing the data.
cd_to_executing_file(__file__)
db_path = '../../data/bloom_1_1B/'
dataset = LLMClassifierDataset(db_path)
features = [item[0] for item in dataset]
labels = [item[1] for item in dataset]
# Tokenizing the features.
features = np.array(tokenizer(features, return_tensors='np', padding='longest'))
# Converting the labels to categorical vectors.
labels = np.array(to_categorical(labels))

# Splitting the data into a training and testing set.
x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.9)


# Loading in the model fitting it to the data.
model = TFAutoModelForSequenceClassification.from_pretrained('facebook/opt-125m', num_labels=2)
model.compile(optimizer=Adam(3e-5),
              metrics=[Accuracy(), AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])
model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=100,
          callbacks=[TensorBoard(log_dir='../../logs/llm_fine_tuning/opt-125m')])
model.save_weights(filepath='opt-125m-fine-tuning.h5', format='h5')

# Testing the model on the test data.
model.load_weights(filepath='opt-125m-fine-tuning.h5')
results = model.evaluate(x_test, y_test, batch_size=32)
