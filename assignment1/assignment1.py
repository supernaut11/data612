import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import Normalization

import re

def get_csv_data(csv_path, cleaner=lambda x: x):
    with open(csv_path, 'r') as input_file:
        return [cleaner(l) for l in input_file.readlines()]

def ingest_csv(csv_data):
    return np.genfromtxt(csv_data, delimiter=',')

def split_x_y(array_in, y_index):
    x = np.hstack((array_in[:, :y_index], array_in[:, y_index+1:]))
    y = array_in[:, y_index]

    return x, y

def build_net(x_train, y_train):
    inputs = keras.Input(shape=(13,))

    x = Normalization()(inputs)
    x = keras.layers.Dense(100, activation="relu")(x)
    x = keras.layers.Dense(100, activation="relu")(x)
    
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x_train, y_train, batch_size=16, epochs=100)

    return model

if __name__ == "__main__":
    data = get_csv_data("housing.csv", lambda x: re.sub(r'( )+', ',', x.strip()))
    data_arr = ingest_csv(data)
    x, y = split_x_y(data_arr, 13)

    model = build_net(x, y)
