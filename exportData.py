import numpy as np
import matplotlib.pyplot as plt
import openpyxl

from Mesure_regression import NSE, R2, MAE, RMSE, MAX_ERROR


def nse(predictions, targets):
    return 1 - (
        np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    )


def printResult(y_test, y_prd, stepDays, folderName):
    print("In", y_test.shape[0], "days:")
    temp1, temp2 = 0, 0
    for i in range(len(y_test)):
        if np.abs(y_test[i] - y_prd[i]) > 14:
            temp1 += 1
        if np.abs(y_test[i] - y_prd[i]) > 18:
            temp2 += 1

    print(
        "{0} days over 14mm ({1:.2f})%".format(
            temp1, float(temp1) / y_test.shape[0] * 100
        )
    )
    print(
        "{0} days over 18mm ({1:.2f})%".format(
            temp2, float(temp2) / y_test.shape[0] * 100
        )
    )

    print("max error", MAX_ERROR(y_test, y_prd)[0])
    print("r2 score:", R2(y_test, y_prd))
    print("nse score:", NSE(y_test, y_prd))
    print("mae score:", MAE(y_test, y_prd))
    print(
        "rmse score:",
        RMSE(y_test, y_prd),
    )

    # export excel result
    input_detail = [
        [
            "callback_days",
            "total_days",
            "over 14",
            "OTR",
            "max error",
            "r2 score",
            "nse score",
            "mae score",
            "rmse score",
        ],
        [
            stepDays,
            y_test.shape[0],
            temp1,
            temp1 / y_test.shape[0],
            MAX_ERROR(y_test, y_prd)[0],
            R2(y_test, y_prd),
            NSE(y_test, y_prd),
            MAE(y_test, y_prd),
            RMSE(y_test, y_prd),
        ],
    ]

    filename = stepDays
    output_excel_path = folderName + "/" + str(filename) + ".csv"
    output_Excel(input_detail, output_excel_path)

    # plotResult(y_test, y_prd, folderName + "/" + str(filename))


def plotResult(y_test, y_prd, imglnk):
    plt.plot(y_test)
    plt.plot(y_prd)
    plt.savefig(imglnk)
    # plt.show()


def output_Excel(input_detail, output_excel_path):
    row = len(input_detail)
    column = len(input_detail[0])

    wb = openpyxl.Workbook()
    ws = wb.active

    for i in range(0, row):
        for j in range(0, column):
            v = input_detail[i][j]
            ws.cell(column=j + 1, row=i + 1, value=v)

    wb.save(output_excel_path)
