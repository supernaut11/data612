import numpy as np
np.random.seed(42)

from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras.datasets.imdb import load_data
from tensorflow.keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D

def get_data():
    return load_data()

def filter_by_top_n(x, n):
    output = np.empty((x.shape[0],), dtype=np.object)

    for idx, value in enumerate(x):
        output[idx] = [v for v in value if v <= n]

    return output

def get_max_review_length(x_data):
    return max([len(x) for x in x_data])

def pad_input_data(x, new_length):
    x_new = np.empty((x.shape[0], new_length), dtype=np.int32)

    for idx, x_cur in enumerate(x):
        x_new[idx,:] = np.pad(np.array(x_cur), (0, new_length - len(x_cur)))

    return x_new

def build_net(x_train, y_train):
    inputs = keras.Input(shape=(x_train.shape[1],))
    
    x = Embedding(5001, 8, mask_zero=True, input_length=x_train.shape[1])(inputs)
    
    x = Conv1D(4, 2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Flatten()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=3)

    model.summary()

    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = filter_by_top_n(x_train, 5000)
    x_test = filter_by_top_n(x_test, 5000)

    max_train_len = get_max_review_length(x_train)
    max_test_len = get_max_review_length(x_test)
    max_review = max(max_train_len, max_test_len)
    print(f'max review length is {max_review}')

    x_train = pad_input_data(x_train, max_review)
    x_test = pad_input_data(x_test, max_review)

    model = build_net(x_train, y_train)

    print(f'making predictions on test set...')
    y_predict = tf.math.greater(model.predict(x_test), 0.5)

    accuracy = accuracy_score(y_test, y_predict)
    print(f'accuracy = {accuracy * 100:.2f}%')
