from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Bidirectional
from pmdarima.arima import auto_arima


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
            300,
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
            300,
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


def getAutoArima(x_train, y_train):
    model = auto_arima(
        x_train.iloc[:, 2].values.tolist(),
        start_p=1,
        start_q=1,
        test="adf",
        max_p=5,
        max_q=5,
        m=1,
        d=1,
        seasonal=False,
        start_P=0,
        D=None,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model
