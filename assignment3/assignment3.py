import numpy as np
from string import ascii_uppercase
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical

ALPHA_TO_NUM = {alpha: idx for idx, alpha in enumerate(ascii_uppercase)}
NUM_TO_ALPHA = {idx: alpha for idx, alpha in enumerate(ascii_uppercase)}

def create_dataset(vals, win_size):
    dataset = [(tuple(vals[i:i+win_size]), vals[i+win_size]) for i in range(len(vals) - win_size)]
    dataset = list(zip(*dataset))

    return dataset[0], dataset[1]

def build_net(x_train, y_train):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]))(inputs)
    outputs = Dense(y_train.shape[1], activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=1, epochs=250)

    return model

window_size = 3
x, y = create_dataset(sorted(NUM_TO_ALPHA.keys()), window_size)

x_win = np.reshape(x, (len(x), window_size, 1)) / float(len(x))
y_win = to_categorical(y)
model = build_net(x_win, y_win)
model.summary()

scores = model.evaluate(x_win, y_win)
print(f"accuracy = {scores[1]*100:.2f}%")

y_predict = model.predict(x_win)
for idx, pred in enumerate(y_predict):
    print([NUM_TO_ALPHA[xi] for xi in x[idx]], 'prediction: ', NUM_TO_ALPHA[np.argmax(pred)])
