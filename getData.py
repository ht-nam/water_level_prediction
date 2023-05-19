import numpy as np
import pandas as pd
import math
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler()


def loadData(
    trainFile,
    testFile,
    cols,
    label_col,
    step_days=1,
    callback_days=1,
    water_level=1,
    is_smote=False,
):
    # start read data from file
    train = readData(file=trainFile, cols=cols)
    test = readData(file=testFile, cols=cols)
    # end read data from file

    # get index of label column name
    lbCol_index = train.columns.values.tolist().index(label_col)

    # normalize data
    scaler.fit(train[cols].values)
    y_scaler.min_, y_scaler.scale_ = (
        scaler.min_[lbCol_index],
        scaler.scale_[lbCol_index],
    )

    # merge callback_days before
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

    # over sampling
    if is_smote:
        x_train, y_train = smote(
            x_train=x_train, y_train=y_train, water_level=water_level
        )

    # shuffle data
    x_train, y_train, x_test, y_test = shuffleData(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test


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
        last_day = x - step_days
        x_train.append(scaled_data[last_day + 1 - callback_days : last_day + 1, :])
        y_train.append(scaled_data[x, lbCol_index])

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train.reshape(-1, 1)


def smote(x_train, y_train, water_level=0):
    z = []
    tr_shape = x_train.shape
    c1, c2 = 0, 0
    for i in range(x_train.shape[0]):
        if y_scaler.inverse_transform(y_train[i][0].reshape(-1, 1)) >= water_level:
            z.append(1)
            c1 += 1
        else:
            z.append(0)
            c2 += 1

    x_train = x_train.reshape(x_train.shape[0], -1)
    data_train = np.concatenate((x_train, y_train), axis=1)
    z = np.array(z)
    k_neighbors = math.ceil(sum(z) * 0.01)
    x_t1, x_t2 = SMOTE(sampling_strategy=1, k_neighbors=k_neighbors).fit_resample(
        data_train, z
    )
    return x_t1[:, :-1].reshape(-1, tr_shape[1], tr_shape[2]), x_t1[:, -1].reshape(
        -1, 1
    )


def shuffleData(x_train, y_train, x_test, y_test):
    trShape, teShape = x_train.shape, x_test.shape

    x_train = x_train.reshape(x_train.shape[0], -1)
    data_train = np.concatenate((x_train, y_train), axis=1)

    x_test = x_test.reshape(x_test.shape[0], -1)
    data_test = np.concatenate((x_test, y_test), axis=1)

    xTr, xTe = train_test_split(
        np.concatenate((data_train, data_test), axis=0), shuffle=True, train_size=0.8
    )

    return (
        xTr[:, :-1].reshape(-1, trShape[1], trShape[2]),
        xTr[:, -1].reshape(-1, 1),
        xTe[:, :-1].reshape(-1, teShape[1], teShape[2]),
        xTe[:, -1].reshape(-1, 1),
    )
