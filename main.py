from lstmModel import smLSTM
from getData import loadData, y_scaler
from exportData import printResult


def main():
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
    model = smLSTM(
        x_train=x_train, y_train=y_train, epochs=epochs, batch_size=batch_size
    )

    # predict
    y_prd = model.predict(x_test)
    y_test_inverse = y_scaler.inverse_transform(y_test)
    y_prd_inverse = y_scaler.inverse_transform(y_prd)
    printResult(y_test_inverse, y_prd_inverse, [], callback_days)


if __name__ == "__main__":
    main()
