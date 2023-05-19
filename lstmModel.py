from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from getData import loadData, y_scaler
from exportData import printResult

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
callbackTime = 6
stepTime = 6
epochs = 10
batchSize = 30
waterLevel = 1.2
isSmote = True


def lstmModel(
    trainFile,
    testFile,
    knowCols,
    labelCol,
    callbackTime=6,
    stepTime=6,
    epochs=100,
    batchSize=30,
    waterLevel=1,
    isSmote=False,
):
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
    model.summary()
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
    printResult(y_test_inverse, y_prd_inverse, [], callbackTime)


lstmModel(
    trainFile,
    testFile,
    knowCols,
    labelCol,
    callbackTime,
    stepTime,
    epochs,
    batchSize,
    waterLevel,
    isSmote,
)
