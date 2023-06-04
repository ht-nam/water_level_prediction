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
from tkinter import messagebox as mb
import re
# # data = [
# #     [573, 0, -7, 0, -22, 0],
# #     [573, 0, -8, 0, -12, 0],
# #     [572, 0.2, -8, 0.2, -1, 0.2],
# #     [572, 0.2, -9, 0.2, 9, 0.2],
# #     [572, 5.8, -10, 5.8, 11, 5.8],
# #     [573, 0.8, -10, 0.8, 14, 0.8],
# # ]

# # data = [
# #     [688, 0, 142, 0, 14, 0],
# #     [687, 0, 141, 0, 22, 0],
# #     [686, 0, 140, 0, 42, 0],
# #     [685, 0, 139, 0, 61, 0],
# #     [684, 0, 138, 0, 81, 0],
# #     [684, 0, 137, 0, 82, 0],
# #     [683, 0, 137, 0, 84, 0],
# #     [683, 3.4, 136, 3.4, 85, 0],
# #     [687, 18.8, 136, 18.8, 69, 0.4],
# #     [690, 0.4, 136, 0.4, 52, 1],
# #     [694, 0, 136, 0, 36, 0],
# #     [697, 0, 136, 0, 37, 2.2],
# # ]

# # print(prd(data, 6))
# # print(prd(dat, 12))


window = tk.Tk()
window.geometry("900x400")

rows = 6 # number of rows
cols = 6 # number of columns
text_var = [] # list of StringVar for each Entry
entries = [] # list of Entry widgets

dtLabel = [
    'WL_KienGiang',
    'RF_KienGiang', 
    'WL_LeThuy', 
    'RF_LeThuy',
    'WL_DongHoi', 
    'RF_DongHoi'
]

def create_entries():
    global rows, cols, text_var, entries
    # loop through the rows and columns
    for i in range(rows):
        # append an empty list to your two arrays
        # so you can append to those later
        text_var.append([])
        entries.append([])
        for j in range(cols):
            # append your StringVar and Entry
            text_var[i].append(tk.StringVar())
            entries[i].append(tk.Entry(window, textvariable=text_var[i][j],width=5))
            entries[i][j].place(x=250 + j*90, y=30 + i*30)

for i in range(len(dtLabel)):
    tk.Label(window,
                  text = dtLabel[i]).place(x = 230 + i * 90,
                                           y = 10) 
    print(dtLabel[i])

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

def get_mat():
    global rows, cols, text_var
    matrix = []
    for i in range(rows):
        matrix.append([])
        for j in range(cols):
            if(text_var[i][j].get() == ''):
                mb.showerror("Thông báo", "Trường dữ liệu không được bỏ trống")
                return
            if(not re.match("^[+-]?(\d+|\d+\.\d+)$", text_var[i][j].get())):
                mb.showerror("Thông báo", "Sai trường dữ liệu")
                return
            matrix[i].append(float(text_var[i][j].get()))
    print(matrix)
    print(prd(matrix, float(size_var.get())))

def delete_entries():
    global rows, cols, text_var, entries
    # loop through the rows and columns
    for i in range(rows):
        for j in range(cols):
            # delete your StringVar and Entry
            text_var[i][j].set("")
            entries[i][j].destroy()
    # clear your two arrays
    text_var.clear()
    entries.clear()

def change_size():
    global rows, cols
    # get the value of the radiobutton variable
    value = int(size_var.get())
    # delete the existing entries
    delete_entries()
    # update the number of columns based on the value
    if value == 6:
        cols = 6
        rows = 6
    elif value == 12:
        cols = 6
        rows = 12
    # create new entries with the updated number of columns
    create_entries()

# create a variable for the radiobuttons
size_var = tk.StringVar()
size_var.set(6) # set the default value

# create two radiobuttons to change the size of the matrix
rb1 = tk.Radiobutton(window, text="6x6", variable=size_var, value=6, command=change_size)
rb2 = tk.Radiobutton(window, text="12x6", variable=size_var, value=12, command=change_size)

# place the radiobuttons on the window
rb1.place(x=100, y=100)
rb2.place(x=100, y=150)

# create the initial entries with 6x6 size
create_entries()

button= Button(window,text="Submit", bg='bisque3', width=15, command=get_mat)
button.place(x=100,y=300)

window.mainloop()