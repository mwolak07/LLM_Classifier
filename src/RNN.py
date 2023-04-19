import numpy as np
import tensorflow as tf
from tensorflow import keras

print("preparing to create dataset")


imdb = tf.keras.datasets.imdb #test dataset to make sure the RNN does something :D
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

#padding data
max_seq_length = max(len(seq) for seq in x_train + x_test)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_seq_length)
y_train = y_train.reshape((-1,1))
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_seq_length)
y_test = y_test.reshape((-1,1))

model = tf.keras.Sequential() #super simple model so my laptop can run it an a reasonable amount of time
model.add(keras.layers.Embedding(input_dim = 10000, output_dim = 32, mask_zero = True))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(1, activation='sigmoid'))#classifies output of LSTM

#prepare for training
metrics = ['accuracy']
loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate = .01)

model.compile(loss = loss, metrics=metrics, optimizer=optim)

#train goofy ass model:
results = model.fit(
 x_train, y_train,
 epochs= 2,
 batch_size = 50,
 validation_data = (x_test, y_test)
)

#saving model I hope
model.save('C:/Users/alex/Documents/GitHub/CS_4400_Final_Project/src/models/rnnsimple.keras')

print(np.mean(results.history["val_acc"]))