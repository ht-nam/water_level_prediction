import numpy as np
import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score, mean_squared_error, max_error
from Mesure_regression import NSE, R2, MAE, RMSE, MAX_ERROR


def nse(predictions, targets):
    return 1 - (
        np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    )


def printResult(y_test, y_prd):
    print("max error", MAX_ERROR(y_test, y_prd))
    print("r2 score:", R2(y_test, y_prd))
    print("nse score:", NSE(y_test, y_prd))
    print("mae score:", MAE(y_test, y_prd))
    print(
        "rmse score:",
        RMSE(y_test, y_prd),
    )
    # print("Thuc te          Du doan")
    # for i in range(0, len(y_test)):
    #     print(y_test[i], "\t", y_prd[i])


def plotResult(y_test, y_prd):
    plt.plot(y_test)
    plt.plot(y_prd)
    plt.show()
