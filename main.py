from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from getData import loadData, y_scaler
from printData import printResult, plotResult


def fitModel(x_train, y_train, epochs=15, batch_size=30):
    model = Sequential()
    model.add(
        LSTM(
            128,
            activation="tanh",
            input_shape=(x_train.shape[1], x_train.shape[2]),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.1))
    model.add(LSTM(64, activation="tanh", return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer="adam", loss="mse")
    model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    return model


def main():
    trainFile = "dataset\dataset_rainseason_train_80.csv"
    testFile = "dataset\dataset_rainseason_test_20.csv"
    cols = [
        "RF_KienGiang",
        "RF_LeThuy",
        "RF_DongHoi",
        "WL_LeThuy",
        "WL_KienGiang",
        "WL_DongHoi",
    ]
    label_col = "WL_LeThuy"
    step_days = 1
    callback_days = 10

    x_train, y_train, x_test, y_test = loadData(
        trainFile=trainFile,
        testFile=testFile,
        cols=cols,
        label_col=label_col,
        step_days=step_days,
        callback_days=callback_days,
    )

    model = fitModel(x_train=x_train, y_train=y_train, epochs=80, batch_size=30)
    y_prd = model.predict(x_test)

    y_test_inverse = y_scaler.inverse_transform(y_test)
    y_prd_inverse = y_scaler.inverse_transform(y_prd)

    printResult(y_test_inverse, y_prd_inverse)
    plotResult(y_test_inverse, y_prd_inverse)


if __name__ == "__main__":
    main()
