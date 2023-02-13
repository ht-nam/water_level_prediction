import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def readData(csv):
    data = pd.read_csv(csv)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data = data[
        [
            "Date",
            "RF_KienGiang",
            "RF_LeThuy",
            "RF_DongHoi",
            "WL_LeThuy",
            "WL_KienGiang",
        ]
    ]
    data = data.fillna(data.bfill())
    data.columns = [
        "Date",
        "RF_KienGiang",
        "RF_LeThuy",
        "RF_DongHoi",
        "WL_LeThuy",
        "WL_KienGiang",
    ]
    return data


def loadData(trainFile, testFile):
    train = readData(trainFile)
    test = readData(testFile)
    train_dates = pd.to_datetime(train["Date"])
    test_dates = pd.to_datetime(test["Date"])
    return train, test, train_dates, test_dates


def mergeRecord(train, step_days=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        train[
            [
                "RF_KienGiang",
                "RF_LeThuy",
                "RF_DongHoi",
                "WL_LeThuy",
                "WL_KienGiang",
            ]
        ].values
    )

    x_train, y_train = [], []

    for x in range(step_days, len(scaled_data) - step_days):
        x_train.append(scaled_data[x - step_days : x, :])
        y_train.append(scaled_data[x + 1, 3])

    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train.reshape(-1, 1), scaler


def main():
    train, test, train_dates, test_dates = loadData(
        "dataset\dataset_rainseason_train_80.csv",
        "dataset\dataset_rainseason_test_20.csv",
    )
    x_train, y_train, scaler = mergeRecord(train=train, step_days=2)

    # defind model
    model = Sequential()
    model.add(
        LSTM(
            100,
            activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2]),
            return_sequences=True,
        )
    )
    model.add(LSTM(100, activation="relu", return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # fit the model
    model.fit(
        x_train, y_train, epochs=15, batch_size=30, validation_split=0.1, verbose=1
    )

    # predict
    actual_data = test[
        [
            "RF_KienGiang",
            "RF_LeThuy",
            "RF_DongHoi",
            "WL_LeThuy",
            "WL_KienGiang",
        ]
    ].values
    total_data = pd.concat(
        (
            train[
                [
                    "RF_KienGiang",
                    "RF_LeThuy",
                    "RF_DongHoi",
                    "WL_LeThuy",
                    "WL_KienGiang",
                ]
            ],
            test[
                [
                    "RF_KienGiang",
                    "RF_LeThuy",
                    "RF_DongHoi",
                    "WL_LeThuy",
                    "WL_KienGiang",
                ]
            ],
        ),
        axis=0,
    )
    prd_days = 2
    model_inputs = total_data[len(total_data) - len(test) - prd_days :].values
    model_inputs = scaler.transform(model_inputs)
    y_test = test["WL_LeThuy"]

    # Make Predictions on Test Data
    x_test = []

    for x in range(prd_days, len(model_inputs)):
        x_test.append(model_inputs[x - prd_days : x, :])

    x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred = model.predict(x_test)
    ad = np.array(scaler.transform(actual_data))
    # print(ad)
    # print("pred", pred[:, 0])
    ad[:, 3] = pred[:, 0]
    # print(ad)
    pred = scaler.inverse_transform(ad)
    # print(pred)

    plt.plot(np.array(actual_data)[:, 3])
    plt.plot(np.array(pred)[:, 3])
    plt.show()
    print("Thuc te ________ Du doan")
    for i in range(0, len(actual_data)):
        print(actual_data[i][3], "\t", pred[i][3])


if __name__ == "__main__":
    main()


