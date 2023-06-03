from keras.models import load_model
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tkinter import *
import numpy as np
import pandas as pd
from tkinter import messagebox
# from myModel import myModel
import os
import tkinter as tk
from tkinter import ttk
from threading import Thread
import threading

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

# callback function to get your StringVars
def get_mat():    
    matrix = []
    for i in range(rows):
        matrix.append([])
        for j in range(cols):
            matrix[i].append(int(text_var[i][j].get()))
    print(matrix)
    print(prd(matrix, int(selected_value.get())))


window = Tk()
window.title("Matrix")
window.geometry("650x500+120+120")
window.configure(bg='bisque2')
window.resizable(False, False)



def update_label():
    print(selected_value.get())
    if(selected_value.get() == 6):
        rows, cols = (6,6)
    else:
        rows,cols = (12, 6)
    return rows,cols
# empty arrays for your Entrys and StringVars
text_var = []
entries = []
selected_value = StringVar(None, "12x6")


label = Label(window, text="Choose matrix size:", font=('arial', 10, 'bold'), bg="bisque2").place(x=250, y=5)
r1 = Radiobutton(window, text="6x6", value=6, variable=selected_value, command=update_label).place(x=50, y=20)
r2 = Radiobutton(window, text="12x6", value=12, variable=selected_value, command=update_label).place(x=500, y=20)

Label(window, text="Enter matrix :", font=('arial', 10, 'bold'), bg="bisque2").place(x=50, y=50)

x2 = 0
y2 = 0

rows, cols = update_label()

for i in range(rows):
    # append an empty list to your two arrays
    # so you can append to those later
    text_var.append([])
    entries.append([])
    for j in range(cols):
        # append your StringVar and Entry
        text_var[i].append(StringVar())
        entries[i].append(Entry(window, textvariable=text_var[i][j],width=3))
        entries[i][j].place(x=200 + x2, y=80 + y2)
        x2 += 30
    y2 += 30
    x2 = 0

button= Button(window,text="Submit", bg='bisque3', width=15, command=get_mat)
button.place(x=160,y=300)

# window.mainloop()
data = [
    [573, 0, -7, 0, -22, 0],
    [573, 0, -8, 0, -12, 0],
    [572, 0.2, -8, 0.2, -1, 0.2],
    [572, 0.2, -9, 0.2, 9, 0.2],
    [572, 5.8, -10, 5.8, 11, 5.8],
    [573, 0.8, -10, 0.8, 14, 0.8],
]

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

print(prd(data, 6))
# print(prd(dat, 12))
