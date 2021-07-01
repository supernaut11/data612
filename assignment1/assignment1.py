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

def split_features_values(array_in, y_index):
    # Create separate views for the features and the output values
    x = array_in[:, ~np.isin(np.arange(array_in.shape[1]), y_index)]
    y = array_in[:, y_index]

    return x, y

def build_net(x_train, y_train):
    # Configure the normalization layer based on the training inputs
    normalizer = Normalization()
    normalizer.adapt(x_train)

    # Input layer with number of inputs being equal to number of features
    inputs = keras.Input(shape=(x_train.shape[1],))

    # Use layer to normalize the input
    x = normalizer(inputs)

    # Use two fully connected layers. Number of units is 80, activation is relu.
    # Note, the values of 80 are used based on a heuristic that the number of
    # weights should be approximately (num_samples * num_features). In our case,
    # num_samples = 506, num_features = 13; product = 6578. We approximate this
    # with two layers of size 80 since 80 ** 2 = 6400
    x = keras.layers.Dense(80, activation="relu")(x)
    x = keras.layers.Dense(80, activation="relu")(x)
    
    # The output should be a single value
    outputs = keras.layers.Dense(1)(x)

    # Create a network from the inputs and outputs we defined
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the network and fit it to the training data
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, batch_size=64, epochs=100)

    return model

def handle_fold(x, y, fold_train, fold_test):
    # Create some aliases for the train/test features and values
    x_train = x[fold_train]
    y_train = y[fold_train]
    x_test = x[fold_test]
    y_test = y[fold_test]

    # Train a network model using the training data
    print(f'training fold...')
    model = build_net(x_train, y_train)

    # Make predictions using the model we trained
    print(f'predicting fold...')
    y_predict = model.predict(x_test)

    # Return the MSE for our model's predictions
    return mean_squared_error(y_test, y_predict)

def main():
    # Set tensorflow seed to constant value so results are reproducible
    tf.random.set_seed(42)

    # Ingest housing data and replace instances of space characters with tabs
    data = get_csv_data("housing.csv", lambda x: re.sub(r'( )+', ',', x.strip()))
    
    # Transform data into an array and split up into features and values
    data_arr = ingest_csv(data)
    x, y = split_features_values(data_arr, 13)

    # Create k-folds object where k=5
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    # Train a network for each of the k-folds splits
    results = [mse for mse in map(lambda f: handle_fold(x, y, f[0], f[1]), folds.split(x, y))]
    
    # Display results
    for idx, mse in enumerate(results):
        print(f'fold {idx} MSE -> {mse:.4f}')
    
    print(f'MSE mean = {np.mean(results):.4f}')
    print(f'MSE standard deviation = {np.std(results):.4f}')

if __name__ == "__main__":
    main()
