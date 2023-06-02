from keras.models import load_model
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def prd(data, callbackDays):
    if callbackDays == 6:
        model = load_model("test\my_lstm_model_6.h5")
        scaler = joblib.load("test\scaler6.save")
    elif callbackDays == 12:
        model = load_model("test\my_lstm_model_12.h5")
        scaler = joblib.load("test\scaler12.save")

    a = np.array(data)
    y_scaler = MinMaxScaler()
    y_scaler.min_, y_scaler.scale_ = (scaler.min_[2], scaler.scale_[2])
    a = np.reshape(scaler.transform(a), (1, a.shape[0], a.shape[1]))
    return y_scaler.inverse_transform(model.predict(a))


# data = [
#     [573, 0, -7, 0, -22, 0],
#     [573, 0, -8, 0, -12, 0],
#     [572, 0.2, -8, 0.2, -1, 0.2],
#     [572, 0.2, -9, 0.2, 9, 0.2],
#     [572, 5.8, -10, 5.8, 11, 5.8],
#     [573, 0.8, -10, 0.8, 14, 0.8],
# ]

# data = [
#     [688, 0, 142, 0, 14, 0],
#     [687, 0, 141, 0, 22, 0],
#     [686, 0, 140, 0, 42, 0],
#     [685, 0, 139, 0, 61, 0],
#     [684, 0, 138, 0, 81, 0],
#     [684, 0, 137, 0, 82, 0],
#     [683, 0, 137, 0, 84, 0],
#     [683, 3.4, 136, 3.4, 85, 0],
#     [687, 18.8, 136, 18.8, 69, 0.4],
#     [690, 0.4, 136, 0.4, 52, 1],
#     [694, 0, 136, 0, 36, 0],
#     [697, 0, 136, 0, 37, 2.2],
# ]

# print(prd(data, 6))
# print(prd(dat, 12))
