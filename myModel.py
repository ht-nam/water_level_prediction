from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, Bidirectional
import os
import tensorflow as tf
from getData import loadData, scaler, y_scaler
from exportData import getResult

trainFile = "dataset\Train_data_WL_RF_21_22.csv"
testFile = "dataset\Test_data_WL_RF_21_22.csv"
knowCols = [
    "WL_KienGiang",
    "RF_KienGiang",
    "WL_LeThuy",
    "RF_LeThuy",
    "WL_DongHoi",
    "RF_DongHoi",
    "Tide_DongHoi",
]
labelCol = "WL_LeThuy"
folderName = "MyResult"
callbackTime = 6
stepTime = 6
epochs = 10
batchSize = 30
waterLevel = 1.2
isSmote = False
modelRadio = "1"
runTypeRadio = "1"


def myModel(
    trainFile,
    testFile,
    knowCols,
    labelCol,
    folderName,
    callbackTime=6,
    stepTime=6,
    epochs=100,
    batchSize=30,
    waterLevel=1,
    isSmote=False,
    modelRadio="1",
    runTypeRadio="1",
):
    if runTypeRadio == "2":
        os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    if not os.path.exists(folderName):
        os.makedirs(folderName)

    # get data
    x_train, y_train, x_test, y_test = loadData(
        trainFile=trainFile,
        testFile=testFile,
        cols=knowCols,
        label_col=labelCol,
        step_days=stepTime,
        callback_days=callbackTime,
        water_level=waterLevel,
        is_smote=isSmote,
    )

    # train model
    model = Sequential()
    if modelRadio == "1":
        model.add(
            LSTM(
                128,
                activation="tanh",
                input_shape=(x_train.shape[1], x_train.shape[2]),
                return_sequences=False,
            )
        )
    elif modelRadio == "2":
        model.add(
            SimpleRNN(
                128,
                activation="tanh",
                input_shape=(x_train.shape[1], x_train.shape[2]),
                return_sequences=False,
            )
        )
    elif modelRadio == "3":
        model.add(
            GRU(
                128,
                activation="tanh",
                input_shape=(x_train.shape[1], x_train.shape[2]),
                return_sequences=False,
            )
        )
    elif modelRadio == "4":
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
    # model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_split=0.3,
        verbose=1,
    )
    # end train model

    # predict
    y_prd = model.predict(x_test)
    y_test_inverse = y_scaler.inverse_transform(y_test)
    y_prd_inverse = y_scaler.inverse_transform(y_prd)
    x_test_inverse = inverseXtest(x_test)
    return getResult(
        x_test_inverse,
        y_test_inverse,
        y_prd_inverse,
        callbackTime,
        waterLevel,
        knowCols,
        labelCol,
        folderName,
    )


def inverseXtest(x_test):
    for i in range(0, x_test.shape[0]):
        x_test[i] = scaler.inverse_transform(x_test[i])
    return x_test


# myModel('./Data/Train_data_WL_RF_21_22.csv', './Data/Test_data_WL_RF_21_22.csv', knowCols, labelCol, folderName)
