import openpyxl
import numpy as np
import pandas as pd


def output_Excel(input_detail, output_excel_path):
    # Xác định số hàng và cột lớn nhất trong file excel cần tạo
    row = len(input_detail)
    column = len(input_detail[0])

    # Tạo một workbook mới và active nó
    wb = openpyxl.Workbook()
    ws = wb.active

    # Dùng vòng lặp for để ghi nội dung từ input_detail vào file Excel
    for i in range(0, row):
        for j in range(0, column):
            v = input_detail[i][j]
            ws.cell(column=j + 1, row=i + 1, value=v)

    # Lưu lại file Excel
    wb.save(output_excel_path)


# def main():
#     input_detail = [
#         ["Sản phẩm", "Mã", "Số lượng", "Giá tiền"],
#         ["Áo sơ mi", "1S25H", 1, 23000],
#         ["Quần bò", "3325H", 7, 50000],
#         ["Áo phông", "16G5H", 45, 70000],
#     ]
#     output_excel_path = "result/data/test.csv"
#     output_Excel(input_detail, output_excel_path)


# if __name__ == "__main__":
#     main()
