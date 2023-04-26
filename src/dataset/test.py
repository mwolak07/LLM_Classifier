
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

# Sample dataset
sentences = ['This is good', 'This is bad', 'This is terrible', 'This is great']

# Labels
labels = np.array([1, 0, 0, 1])

# Tokenize the sentences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences
maxlen = max([len(x) for x in sequences])
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Define the model
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=10000, output_dim=16))
model.add(keras.layers.LSTM(units=32))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10)
