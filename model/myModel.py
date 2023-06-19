import os
from getData import loadData, readReferenceValue, scaler, y_scaler
from exportData import getResult
from model.allModel import getLSTM, getRNN, getGRU, getBidirectional, getAutoArima
import numpy as np

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
callbackTime = 12
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
    reference_col="",
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
        modelRadio=modelRadio,
    )

    # print(x_test.shape, y_test.shape)
    refValue = (
        readReferenceValue(testFile, reference_col, callbackTime, stepTime)
        if reference_col != ""
        else ""
    )
    # print(refValue.shape)

    # train model
    model = (
        getLSTM(x_train, y_train)
        if modelRadio == "1"
        else getRNN(x_train, y_train)
        if modelRadio == "2"
        else getGRU(x_train, y_train)
        if modelRadio == "3"
        else getBidirectional(x_train, y_train)
        if modelRadio == "4"
        else getAutoArima(x_train, y_train)
    )

    if modelRadio != "5":
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
        y_prd = model.predict(x_test)
        y_test_inverse = y_scaler.inverse_transform(y_test)
        y_prd_inverse = y_scaler.inverse_transform(y_prd)
        x_test_inverse = inverseXtest(x_test)
    else:
        prediction, _ = model.predict(
            n_periods=x_test.shape[0] - callbackTime - stepTime, return_conf_int=True
        )
        x_test_inverse = np.array(
            x_test.iloc[callbackTime + stepTime :, 2].values
        ).reshape(x_test.shape[0] - callbackTime - stepTime, 1, 1)
        y_prd_inverse = np.array(prediction).reshape(
            x_test.shape[0] - callbackTime - stepTime, 1
        )
        y_test_inverse = np.array(
            x_test.iloc[callbackTime + stepTime :, 2].values
        ).reshape(x_test.shape[0] - callbackTime - stepTime, 1)

    return getResult(
        x_test_inverse,
        y_test_inverse,
        y_prd_inverse,
        callbackTime,
        waterLevel,
        knowCols,
        labelCol,
        folderName,
        refValue,
        reference_col,
        modelRadio,
    )


def inverseXtest(x_test):
    for i in range(0, x_test.shape[0]):
        x_test[i] = scaler.inverse_transform(x_test[i])
    return x_test
