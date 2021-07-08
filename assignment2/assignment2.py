import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D

MAX_WORDS_BOUND = 5001

def build_net(x_train, y_train, n):
    inputs = keras.Input(shape=(x_train.shape[1],))
    x = Embedding(n, 8, mask_zero=True, input_length=x_train.shape[1])(inputs)
    x = Conv1D(4, 2)(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Flatten()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=5)

    return model

# Load data such that we index words from 0 and assign out-of-vocab token outside of core
# vocabulary range. This deals with Keras's odd behavior of treating both 0 _and_ 'index_from'
# (3 by default) as zero, but the Embedding layer wanting only _actual_ zeros.
print('downloading data...')
(x_train, y_train), (x_test, y_test) = load_data(num_words=MAX_WORDS_BOUND,
                                                 oov_char=MAX_WORDS_BOUND,
                                                 index_from=0)

# Drop the first value in training data, it's just the start token and isn't useful here
x_train = [xi[1:] for xi in x_train]
x_test = [xi[1:] for xi in x_test]

y_train = np.array(y_train)
y_test = np.array(y_test)

# Pad any rows that are shorter than the longest review with 0's
print('padding data to maximum review length...')
cols = range(max(max(len(x) for x in x_train), max(len(x) for x in x_test)))
x_train = pd.DataFrame(x_train).reindex(columns=cols).fillna(value=0).astype(np.int32)
x_test = pd.DataFrame(x_test).reindex(columns=cols).fillna(value=0).astype(np.int32)

print('building model...')
model = build_net(x_train, y_train, MAX_WORDS_BOUND + 1)
model.summary()

# Use output of sigmoid to make classification decision
print(f'making predictions on test set...')
y_predict = tf.math.greater(model.predict(x_test), 0.5)

accuracy = accuracy_score(y_test, y_predict)
print(f'accuracy = {accuracy * 100:.2f}%')