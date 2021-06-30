import numpy as np
import re
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization

def get_csv_data(csv_path, cleaner=lambda x: x):
    with open(csv_path, 'r') as input_file:
        return [cleaner(l) for l in input_file.readlines()]

def ingest_csv(csv_data):
    return np.genfromtxt(csv_data, delimiter=',')

def split_features_labels(array_in, y_index):
    x = np.hstack((array_in[:, :y_index], array_in[:, y_index+1:]))
    y = array_in[:, y_index]

    return x, y

def build_net(x_train, y_train):
    normalizer = Normalization()
    normalizer.adapt(x_train)

    inputs = keras.Input(shape=(x_train.shape[1],))

    x = normalizer(inputs)
    x = keras.layers.Dense(80, activation="sigmoid")(x)
    x = keras.layers.Dense(80, activation="sigmoid")(x)
    
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='mse')

    model.fit(x_train, y_train, batch_size=32, epochs=100)

    return model

def handle_fold(x, y, fold_train, fold_test):
    x_train = x[fold_train]
    y_train = y[fold_train]
    x_test = x[fold_test]
    y_test = y[fold_test]

    print(f'training fold...')
    model = build_net(x_train, y_train)

    print(f'predicting fold...')
    y_predict = model.predict(x_test)
    return mean_squared_error(y_test, y_predict)

if __name__ == "__main__":
    tf.random.set_seed(0)

    data = get_csv_data("housing.csv", lambda x: re.sub(r'( )+', ',', x.strip()))
    data_arr = ingest_csv(data)
    x, y = split_features_labels(data_arr, 13)

    folds = KFold(n_splits=5, shuffle=True, random_state=0)

    results = [mse for mse in map(lambda f: handle_fold(x, y, f[0], f[1]), folds.split(x, y))]
    for idx, mse in enumerate(results):
        print(f'fold {idx} MSE -> {mse:.4f}')
    
    print(f'mean MSE = {np.mean(results):.4f}; std MSE = {np.std(results):.4f}')
