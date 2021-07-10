import numpy as np
import pandas as pd
from string import ascii_uppercase
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical

np.random.seed(42)

ALPHABET = ascii_uppercase
ALPHA_TO_NUM = {alpha: idx for idx, alpha in enumerate(ALPHABET)}
NUM_TO_ALPHA = {idx: alpha for idx, alpha in enumerate(ALPHABET)}

def create_dataset(val_mapping, window_size):
    dataset = {}
    keys = [k for k in val_mapping.keys()]
    
    for idx in range(len(val_mapping) - window_size):
        dataset[tuple(keys[idx:idx+window_size])] = idx + window_size

    return tuple(dataset.keys()), tuple(dataset.values())

def build_net(x_train, y_train):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]))(inputs)
    outputs = Dense(y_train.shape[1], activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=1, epochs=500)

    return model

def main():
    window_size = 1
    x, y = create_dataset(NUM_TO_ALPHA, window_size)
    x = np.reshape(x, (len(x), window_size, 1)) / float(len(x))
    y = to_categorical(y)

    model = build_net(x, y)
    model.summary()

    scores = model.evaluate(x, y)
    print(f"accuracy = {scores[1]*100:.2f}%")

if __name__ == "__main__":
    main()