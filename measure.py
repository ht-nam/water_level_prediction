import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def R2(y_test, y_pred):
    y_test_mean = np.mean(y_test)
    #print(y_test_mean)
    y_pred_mean = np.mean(y_pred)
    #print(y_pred_mean)
    S_res = np.sum((y_test - y_test_mean)*(y_pred - y_pred_mean))
    #print(S_res)
    S_tot1 = np.sum((y_test - y_test_mean)**2)
    #print(S_tot1)
    S_tot2 = np.sum((y_pred - y_pred_mean)**2)
    #print(S_tot2)
    S_tot = np.sqrt(S_tot1 * S_tot2)
    #print(S_tot1 * S_tot2, S_tot)
    return (S_res / S_tot) ** 2

def MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)

def MAX_ERROR(y_test, y_pred):
    max = 0
    for i in range(0, len(y_test)):
        if (max < abs(y_test[i] - y_pred[i])):
            max = abs(y_test[i] - y_pred[i])
    return max