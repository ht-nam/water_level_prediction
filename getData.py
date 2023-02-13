import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler()


def readData(file, cols):
    data = pd.read_csv(file)
    data = data[cols]
    data = data.fillna(data.bfill())
    data.columns = cols
    return data


def mergeRecord(data, cols, lbCol_index, step_days=1, callback_days=1):
    data = data[cols].values
    scaled_data = scaler.transform(data)

    x_train, y_train = [], []
    for x in range(callback_days - 1 + step_days, len(data)):
        # last day of x_train item
        last_day = x - step_days

        x_train.append(scaled_data[last_day + 1 - callback_days : last_day + 1, :])
        # x_train.append(data[last_day + 1 - callback_days : last_day + 1, :])
        y_train.append(scaled_data[x, lbCol_index])
        # y_train.append(data[x, lbCol_index])

    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train.reshape(-1, 1)


def loadData(trainFile, testFile, cols, label_col, step_days=1, callback_days=1):
    train = readData(file=trainFile, cols=cols)
    test = readData(file=testFile, cols=cols)
    lbCol_index = train.columns.values.tolist().index(label_col)

    scaler.fit(train[cols].values)
    y_scaler.min_, y_scaler.scale_ = (
        scaler.min_[lbCol_index],
        scaler.scale_[lbCol_index],
    )

    x_train, y_train = mergeRecord(
        data=train,
        cols=cols,
        lbCol_index=lbCol_index,
        step_days=step_days,
        callback_days=callback_days,
    )
    x_test, y_test = mergeRecord(
        data=test,
        cols=cols,
        lbCol_index=lbCol_index,
        step_days=step_days,
        callback_days=callback_days,
    )
    return x_train, y_train, x_test, y_test


def __test():
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
    step_days = 2
    callback_days = 5

    x_train, y_train, x_test, y_test = loadData(
        trainFile=trainFile,
        testFile=testFile,
        cols=cols,
        label_col=label_col,
        step_days=step_days,
        callback_days=callback_days,
    )

    # print(y_train.shape, y_test.shape)
    # print("ytrain", y_train, "\nytest", y_test)
    # print(x_train.shape, x_test.shape)
    # print("ytrain", x_train, "\nytest", x_test)
    # print("b", y_test)
    # print("a", y_scaler.inverse_transform(y_test))


#
# __test()
