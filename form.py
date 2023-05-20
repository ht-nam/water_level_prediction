from tkinter import *
import numpy as np
# from linear_regression import LR_Model
import pandas as pd
from tkinter import messagebox
# import lstmModel
# import sys
# sys.path.insert(1, '../lstmModel.py')
from lstmModel import lstmModel


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
    file_name = textbox_file_name.get()
    file_name = file_name.strip()

    X = pd.read_excel(file_name)
    X = np.array(X.values)

    for i in range(X.shape[0]):
        print('Kich ban', i)
        foldername = X[i][1]
        print('foldername', foldername)
        file_train = X[i][2]
        file_test = X[i][3]
        max_numdays = X[i][4]
        print("max_numdays", max_numdays)
        max_afterdays = X[i][5]
        print("max_afterdays", max_afterdays)
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

        pred_attribute = split_string(X[i][8])
        print('pred_attribute', pred_attribute)

        if (X[i][9] != X[i][9]):
            threshold = 999999.0
        else:
            threshold = float(X[i][9])
        print('threshold', threshold)

        if (X[i][10] == 0):
            nomalize = False
        else:
            nomalize = True
        print('nomalize', nomalize)

        if (X[i][11] == 0):
            smote = False
        else:
            smote = True
        print('smote', smote)

        if (X[i][12] != X[i][12]):
            smote_threshold = -999999.0
        else:
            smote_threshold = float(X[i][12])
        print('threshold_smote', smote_threshold)
        if(X[i][13] != X[i][13]):
            epochs = 999999.0
        else:
            epochs = X[i][13]
        print('number of epoch', epochs)
        if(X[i][14] != X[i][14]):
            batch_size = 999999.0
        else:
            batch_size = X[i][14]
        print('number of batch size', batch_size)

        # lstmModel(trainFile=file_train, testFile= file_test, know_attributes= know_attributes, knowCols=unknow_attributes, 
        # labelCol = pred_attribute, callbackTime=max_afterdays, stepTime = max_numdays, 
        # epochs=epochs, batchSize=batch_size, waterLevel=threshold,isSmote = smote, foldername=foldername)
        # LR_Model(file_train, file_test, know_attributes, unknow_attributes, pred_attribute, foldername, max_numdays,
        #          max_afterdays, threshold, nomalize, smote, smote_threshold)
    messagebox.showinfo("Notification", "Finished testing")



''''
    file_test = textbox_file_test.get()
    file_test = file_test.strip()

    text_know_attributes = textbox_know_attributes.get()
    know_attributes = split_string(text_know_attributes)

    text_unknow_attributes = textbox_unknow_attributes.get()
    unknow_attributes = split_string(text_unknow_attributes)

    pred_attribute = textbox_pred_attributes.get()
    pred_attribute = split_string(pred_attribute)

    foldername = textbox_folder_name.get()
    foldername = foldername.strip()

    max_numdays = int(textbox_numdays.get())
    max_afterdays = int(textbox_afterdays.get())

    LR_Model(file_train, file_test, know_attributes, unknow_attributes, pred_attribute, foldername, max_numdays,
             max_afterdays, nomalize=False)'''
form = Tk()
form.title("Thực nghiệm mô hình Hồi quy tuyến tính:")
form.geometry("1000x500")

lable_file_name = Label(form, text = "Kịch bản thực nghiệm:")
lable_file_name.grid(row = 1, column = 1, padx = 40, pady = 10, sticky=W)
textbox_file_name = Entry(form, width=50)
textbox_file_name.grid(row = 1, column = 2)

button_submit = Button(form, text = 'Submit', width=10, command = getform)
button_submit.grid(row = 9, column = 2, pady = 20, sticky=W)

button_reset = Button(form,text='Reset', width=10, command=lambda:reset_form())
button_reset.grid(row=9, column=3, pady=20, sticky=W)

form.mainloop()

# # Create a tkinter window
# window = tk.Tk()

# # Create a label to show the loading message
# label = tk.Label(window, text="Loading...")
# label.pack()

# # Define a function to fetch data
# def fetch_data():
#     # Simulate fetching data with time.sleep
#     time.sleep(5)
#     # Change the label text after fetching data
#     label.config(text="Data fetched!")

# # Define a function to start a thread for fetching data
# def start_thread():
#     # Create a thread object with the fetch_data function as target
#     thread = threading.Thread(target=fetch_data)
#     # Start the thread
#     thread.start()


# import tkinter as tk
# from PIL import Image, ImageTk
# import threading
# import time

# # Create a tkinter window
# window = tk.Tk()

# # Load an animated GIF image of a circular loading
# image = Image.open("loading.gif")
# frames = ImageTk.PhotoImage(image)

# # Create a label to display the image
# label = tk.Label(window, image=frames)
# label.pack()

# # Define a function to update the image frame by frame
# def update(ind):
#     # Get the next frame of the image
#     frame = frames[ind]
#     # Update the label image
#     label.configure(image=frame)
#     # Increment the index and reset if it reaches the end
#     ind += 1
#     if ind == frames.n_frames:
#         ind = 0
#     # Schedule the next update after 100 ms
#     window.after(100, update, ind)

# # Define a function to fetch data
# def fetch_data():
#     # Simulate fetching data with time.sleep
#     time.sleep(5)
#     # Destroy the window after fetching data
#     window.destroy()

# # Start a thread for fetching data
# thread = threading.Thread(target=fetch_data)
# thread.start()

# # Start updating the image from the first frame
# window.after(0, update, 0)

# # Start the main loop
# window.mainloop()