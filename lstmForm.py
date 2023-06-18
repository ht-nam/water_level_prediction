from tkinter import *
import numpy as np
import pandas as pd
from tkinter import messagebox
from model.myModel import myModel
import os
import tkinter as tk
from tkinter import ttk
from threading import Thread
import threading
from exportData import output_Excel


def reset_form():
    for widget in form.winfo_children():
        if isinstance(widget, Entry):  # If this is an Entry widget class
            widget.delete(0, "end")
    pb["value"] = 0
    value_label["text"] = ""
    modelRadio.set(1)
    runTypeRadio.set(1)


def split_string(s):
    str = s.split(",")
    result = []
    for i in range(len(str)):
        result.append(str[i].strip())
    return result


def job():
    file_name = "./Kichban/" + textbox_file_name.get()
    file_name = file_name.strip()

    try:
        X = pd.read_excel(file_name)
    except:
        try:
            X = pd.read_excel(textbox_file_name.get())
        except:
            messagebox.showerror("ERROR", message="File not found error!")
    X = np.array(X.values)

    for i in range(X.shape[0]):
        print("Kich ban", i)
        foldername = X[i][1]
        # print("foldername", foldername)
        file_train = X[i][2]
        file_test = X[i][3]
        max_numdays = X[i][4]
        max_afterdays = X[i][5]
        if X[i][7] != X[i][7]:
            know_attributes = []
        else:
            know_attributes = split_string(X[i][7])
        # print("know_attributes", know_attributes)

        if X[i][8] != X[i][8]:
            unknow_attributes = []
        else:
            unknow_attributes = split_string(X[i][8])
        # print("unknow_attributes", unknow_attributes)

        if X[i][9] != X[i][9]:
            threshold = 999999.0
        else:
            threshold = float(X[i][9])
        # print("threshold", threshold)

        if X[i][10] == 0:
            smote = False
        else:
            smote = True
        # print("smote", smote)

        if X[i][11] != X[i][11]:
            smote_threshold = -999999.0
        else:
            smote_threshold = float(X[i][11])
        # print("threshold_smote", smote_threshold)
        if X[i][12] != X[i][12]:
            epochs = 999999.0
        else:
            epochs = X[i][12]
        # print("number of epoch", epochs)
        if X[i][13] != X[i][13]:
            batch_size = 999999.0
        else:
            batch_size = X[i][13]
        # print("number of batch size", batch_size)
        if X[i][14] != X[i][14]:
            reference_col = ""
        else:
            reference_col = X[i][14]
        # print(reference_col)

        input_detail = [
            [
                "callback_days",
                "total_days",
                "over_days",
                "OTR",
                "max error",
                "r2 score",
                "nse score",
                "mae score",
                "rmse score",
            ]
        ]

        # for j in range(1, max_numdays + 1):
        for j in range(max_numdays, max_numdays + 1):
            input_detail.append(
                myModel(
                    file_train,
                    file_test,
                    know_attributes,
                    unknow_attributes[0],
                    foldername,
                    j,
                    max_afterdays,
                    epochs,
                    batch_size,
                    threshold,
                    smote,
                    reference_col,
                    modelRadio.get(),
                    runTypeRadio.get(),
                )
            )
            progress(X.shape[0] * max_numdays)
        output_Excel(input_detail, foldername + "/summary.xlsx")

    # stop()
    messagebox.showinfo("Notification", "Finished testing")
    # button_submit["state"] = NORMAL
    # button_reset["state"] = NORMAL


def progress(length):
    if length:
        pb["value"] += 100 / length
        value_label["text"] = update_progress_label()
    # else:
    #     messagebox.showinfo(message='The progress completed!')


def update_progress_label():
    return f"Current Progress: {round(pb['value'])}%"


def stop():
    pb.stop()
    value_label["text"] = update_progress_label()


def threading():
    # button_submit["state"] = DISABLED
    # button_reset["state"] = DISABLED
    trainThread = Thread(target=job)
    trainThread.start()
    # trainThread.join()


def formConfig():
    form.title("Thực nghiệm mô hình")
    form.geometry("1000x500")
    # outside form
    folder = "./Kichban/"
    files = [f for f in os.listdir(folder) if f.endswith(".xlsx") or f.endswith(".csv")]

    lable_file_name = Label(form, text="Kịch bản thực nghiệm:")
    lable_file_name.grid(row=8, column=1, padx=40, pady=10, sticky=W)
    textbox_file_name = ttk.Combobox(form, width=50, values=files)
    textbox_file_name.grid(row=8, column=2)

    button_submit = Button(form, text="Submit", width=10, command=threading)
    button_submit.grid(row=10, column=2, pady=20, sticky=W)

    button_reset = Button(form, text="Reset", width=10, command=lambda: reset_form())
    button_reset.grid(row=10, column=3, pady=20, sticky=W)

    # radio button
    selected = tk.StringVar()
    cpuRB = ttk.Radiobutton(form, text="CPU", value=1, variable=selected)
    cpuRB.grid(row=9, column=2, sticky=W)
    gpuRB = ttk.Radiobutton(form, text="GPU", value=2, variable=selected)
    gpuRB.grid(row=9, column=3, sticky=W)
    selected.set(1)

    radioSelected = tk.StringVar()
    lstmRB = ttk.Radiobutton(form, text="LSTM", value=1, variable=radioSelected)
    lstmRB.grid(row=8, padx=10, column=3, sticky=W)
    rnnRB = ttk.Radiobutton(form, text="RNN", value=2, variable=radioSelected)
    rnnRB.grid(row=8, padx=10, column=4, sticky=W)
    gruRB = ttk.Radiobutton(form, text="GRU", value=3, variable=radioSelected)
    gruRB.grid(row=8, padx=10, column=5, sticky=W)
    biRB = ttk.Radiobutton(form, text="BiDirectional", value=4, variable=radioSelected)
    biRB.grid(row=8, padx=10, column=6, sticky=W)
    arRB = ttk.Radiobutton(form, text="Arima", value=5, variable=radioSelected)
    arRB.grid(row=8, padx=10, column=7, sticky=W)
    radioSelected.set(1)
    # radio button

    # place the progressbar
    pb = ttk.Progressbar(
        form, orient="horizontal", mode="determinate", length=280, maximum=100
    )
    pb.grid(column=2, row=12, columnspan=2, padx=10, pady=20)
    # label
    value_label = ttk.Label(form, text="")
    value_label.grid(column=2, row=13, columnspan=2)

    return (
        lable_file_name,
        textbox_file_name,
        button_submit,
        button_reset,
        pb,
        value_label,
        radioSelected,
        selected,
    )


form = Tk()
trainThread = Thread()
(
    lable_file_name,
    textbox_file_name,
    button_submit,
    button_reset,
    pb,
    value_label,
    modelRadio,
    runTypeRadio,
) = formConfig()

form.mainloop()
