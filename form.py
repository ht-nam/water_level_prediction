from tkinter import *
import numpy as np
# from linear_regression import LR_Model
import pandas as pd
from tkinter import messagebox
# import lstmModel
# import sys
# sys.path.insert(1, '../lstmModel.py')
from lstmModel import lstmModel
import os
import tkinter as tk
from tkinter import ttk


def reset_form():
    for widget in form.winfo_children():
        if isinstance(widget, Entry):  # If this is an Entry widget class
            widget.delete(0, 'end')


def split_string(s):
    str = s.split(",")
    result = []
    for i in range(len(str)):
        result.append(str[i].strip())
    return result

def getform():
    file_name = './Kichban/' + textbox_file_name.get()
    file_name = file_name.strip()

    X = pd.read_excel(file_name)
    X = np.array(X.values)

    for i in range(X.shape[0]):
        print('Kich ban', i)
        foldername = X[i][1]
        print('foldername', foldername)
        file_train = X[i][2]
        file_test = X[i][3]
        max_numdays = X[i][4] + 1
        max_afterdays = X[i][5]
        if (X[i][6] != X[i][6]):
            know_attributes = []
        else:
            know_attributes = split_string(X[i][6])
        print('know_attributes', know_attributes)

        if (X[i][7] != X[i][7]):
            unknow_attributes = []
        else:
            unknow_attributes = split_string(X[i][7])
        print('unknow_attributes', unknow_attributes)

        if (X[i][8] != X[i][8]):
            threshold = 999999.0
        else:
            threshold = float(X[i][8])
        print('threshold', threshold)

        if (X[i][9] == 0):
            smote = False
        else:
            smote = True
        print('smote', smote)

        if (X[i][10] != X[i][10]):
            smote_threshold = -999999.0
        else:
            smote_threshold = float(X[i][12])
        print('threshold_smote', smote_threshold)
        if(X[i][11] != X[i][11]):
            epochs = 999999.0
        else:
            epochs = X[i][11]
        print('number of epoch', epochs)
        if(X[i][12] != X[i][12]):
            batch_size = 999999.0
        else:
            batch_size = X[i][12]
        print('number of batch size', batch_size)
        progress(X.shape[0])
        update_progress_label()
        # lstmModel(trainFile=file_train, testFile= file_test, know_attributes= know_attributes, knowCols=unknow_attributes, 
        # labelCol = pred_attribute, callbackTime=max_afterdays, stepTime = max_numdays, 
        # epochs=epochs, batchSize=batch_size, waterLevel=threshold,isSmote = smote, foldername=foldername)
    # stop()
    messagebox.showinfo("Notification", "Finished testing")


def update_progress_label():
    return f"Current Progress: {pb['value']}%"


def progress(length):
    if length:
        pb['value'] += 100 / length
        value_label['text'] = update_progress_label()
    # else:
    #     messagebox.showinfo(message='The progress completed!')


def stop():
    pb.stop()
    value_label['text'] = update_progress_label()

# form instantiate
form = Tk()
form.title("Thực nghiệm mô hình Hồi quy tuyến tính:")
form.geometry("1000x500")
# outside form
folder = r"C:\Users\tavan\OneDrive\Documents\Attendance\Python\water_level_prediction\Kichban"
files = [f for f in os.listdir(folder) if f.endswith(".xlsx") or f.endswith('.csv')]
# 
# form content


lable_file_name = Label(form, text = "Kịch bản thực nghiệm:")
lable_file_name.grid(row = 8, column = 1, padx = 40, pady = 10, sticky=W)
textbox_file_name = ttk.Combobox(form, width=50, values=files)
textbox_file_name.grid(row = 8, column = 2)

button_submit = Button(form, text = 'Submit', width=10, command = getform)
button_submit.grid(row = 10, column = 2, pady = 20, sticky=W)

button_reset = Button(form,text='Reset', width=10, command=lambda:reset_form())
button_reset.grid(row=10, column=3, pady=20, sticky=W)

# value_label = ttk.Label(form, text=update_progress_label())
# value_label.grid(column=1, row=12, columnspan=2)
# progressbar
pb = ttk.Progressbar(
    form,
    orient='horizontal',
    mode='determinate',
    length=280,
    maximum=100
)
# place the progressbar
pb.grid(column=2, row=12, columnspan=2, padx=10, pady=20)
# label
value_label = ttk.Label(form, text=update_progress_label())
value_label.grid(column=2, row=13, columnspan=2)

# form content
form.mainloop()

