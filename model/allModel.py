from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Bidirectional


def getLSTM(x_train, y_train):
    model = Sequential()
    model.add(
        LSTM(
            128,
            activation="tanh",
            input_shape=(x_train.shape[1], x_train.shape[2]),
            return_sequences=False,
        )
    )
    model.add(Dense(y_train.shape[1], activation="tanh")),
    model.compile(optimizer="adam", loss="mse")
    return model


def getRNN(x_train, y_train):
    model = Sequential()
    model.add(
        SimpleRNN(
            128,
            activation="tanh",
            input_shape=(x_train.shape[1], x_train.shape[2]),
            return_sequences=False,
        )
    )
    model.add(Dense(y_train.shape[1], activation="tanh")),
    model.compile(optimizer="adam", loss="mse")
    return model


def getGRU(x_train, y_train):
    model = Sequential()
    model.add(
        GRU(
            128,
            activation="tanh",
            input_shape=(x_train.shape[1], x_train.shape[2]),
            return_sequences=False,
        )
    )
    model.add(Dense(y_train.shape[1], activation="tanh")),
    model.compile(optimizer="adam", loss="mse")
    return model


def getBidirectional(x_train, y_train):
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(
                128,
                # activation="tanh",
                kernel_initializer="he_normal",
                input_shape=(x_train.shape[1], x_train.shape[2]),
                return_sequences=False,
            )
        )
    )
    model.add(Dense(y_train.shape[1], activation="tanh")),
    model.compile(optimizer="adam", loss="mse")
    return model
