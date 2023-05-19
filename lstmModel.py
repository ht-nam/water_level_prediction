from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from getData import loadData, y_scaler
from exportData import printResult

trainFile = "dataset\Train_data_WL_RF_21_22.csv"
testFile = "dataset\Test_data_WL_RF_21_22.csv"
cols = [
    "WL_KienGiang",
    "RF_KienGiang",
    "WL_LeThuy",
    "RF_LeThuy",
    "WL_DongHoi",
    "RF_DongHoi",
    "Tide_DongHoi",
]
is_smote = True
label_col = "WL_LeThuy"
step_days = 6
callback_days = 6
epochs = 10
batch_size = 30
water_level = 1.2


def lstmModel():
    # get data
    x_train, y_train, x_test, y_test = loadData(
        trainFile=trainFile,
        testFile=testFile,
        cols=cols,
        label_col=label_col,
        step_days=step_days,
        callback_days=callback_days,
        water_level=water_level,
        is_smote=is_smote,
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
        batch_size=batch_size,
        validation_split=0.3,
        verbose=1,
    )
    # end train model

    # predict
    y_prd = model.predict(x_test)
    y_test_inverse = y_scaler.inverse_transform(y_test)
    y_prd_inverse = y_scaler.inverse_transform(y_prd)
    printResult(y_test_inverse, y_prd_inverse, [], callback_days)


lstmModel()
