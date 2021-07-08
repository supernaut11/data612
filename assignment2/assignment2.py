import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D

# Setting n=5003 accounts for the 5000 top tokens, plus the out-of-vocabulary char (5001), 
# plus the padding char (5002), plus additional offset of 1 per Embedding API docs
def build_net(x_train, y_train, n=5003):
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

print('downloading data...')
# num_words = 5003 so that we get the top 5000 most frequent words. Without doing this,
# the start_char (1) and the oov_char (2) from load_data prevent us from getting 5000 tokens
(x_train, y_train), (x_test, y_test) = load_data(num_words=5003)
y_train = np.array(y_train)
y_test = np.array(y_test)

print('padding data to maximum review length...')
cols = range(max(max(len(x) for x in x_train), max(len(x) for x in x_test)))
x_train = pd.DataFrame([xi[1:] for xi in x_train]).reindex(columns=cols).fillna(value=0)
x_test = pd.DataFrame([xi[1:] for xi in x_test]).reindex(columns=cols).fillna(value=0)

print('building model...')
model = build_net(x_train, y_train)
model.summary()

print(f'making predictions on test set...')
y_predict = tf.math.greater(model.predict(x_test), 0.5)

accuracy = accuracy_score(y_test, y_predict)
print(f'accuracy = {accuracy * 100:.2f}%')