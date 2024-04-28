import os
import pandas as pd
from fuzzywuzzy import process
from gapi import apiResult
from tallyYolo import predict_and_show_labels
from resnt_test import resnetPred


def process_files(base_path):
    results = {}  # Dictionary to hold results organized by pages, rows, and columns
    data_for_excel = []  # List to hold data for Excel export

    # Get all page folders and sort them numerically by extracting numbers
    page_folders = [f for f in os.listdir(base_path) if f.startswith("page_")]
    page_folders_sorted = sorted(
        page_folders, key=lambda x: int(x.split('_')[1]))

    for page_folder in page_folders_sorted:
        page_number = page_folder.split('_')[1]
        print("PAGE NUM "+page_number)
        page_path = os.path.join(base_path, page_folder)
        results[page_number] = results.get(page_number, {})

        row_folders = [f for f in os.listdir(
            page_path) if f.startswith("row_")]
        row_folders_sorted = sorted(
            row_folders, key=lambda x: int(x.split('_')[1]))

        for row_folder in row_folders_sorted:
            row_number = row_folder.split('_')[1]
            print("ROW NUM "+row_number)
            row_path = os.path.join(page_path, row_folder)
            results[page_number][row_number] = results[page_number].get(
                row_number, {})
            row_data = []  # Initialize list to store row data for Excel

            col_folders = sorted(os.listdir(row_path),
                                 key=lambda x: int(x.split('_')[1]))

            for col_folder in col_folders:
                col_number = col_folder.split('_')[1]
                col_path = os.path.join(row_path, col_folder)
                sort_file = sorted(os.listdir(col_path),
                                   key=lambda x: int(x.split('_')[0]))

                # col_path = os.path.join(row_path, col_folder)
                results[page_number][row_number][col_number] = []

                for i in range(len(sort_file)):
                    file_name = sort_file[i]
                    file_path = os.path.join(col_path, file_name)
                    result = None
                    if 'column_1' == col_folder:
                        if 'N_A' in file_name or 'N-A' in file_name:
                            result = 'N/A'
                        elif 'Number' in file_name:
                            result = apiResult(file_path)
                    elif 'column_2' == col_folder:
                        if 'N_A' in file_name or 'N-A' in file_name:
                            result = 'N/A'
                        elif 'Words' in file_name:
                            result = apiResult(file_path)
                    elif 'column_3' == col_folder:
                        if 'N_A' in file_name or 'N-A' in file_name:
                            result = 'N/A'
                        elif 'Circled_Number' in file_name:
                            continue
                        elif 'Number' in file_name:
                            result = apiResult(file_path)

                    elif col_folder == 'column_4':
                        # if page_number == "10" and row_number == "3":
                        #     a = 0
                        if 'N_A' in file_name or 'N-A' in file_name:
                            result = 'N/A'
                        elif 'Words-and-tallys' in file_name:
                            result = predict_and_show_labels(file_path)
                        elif 'Words' in file_name:
                            result = apiResult(file_path)
                        elif 'Circled_Number' in file_name:
                            continue
                        elif 'Number' in file_name:
                            result = apiResult(file_path)
                            if result is None or result is 'None':
                                result = '0'

                    if result:
                        results[page_number][row_number][col_number].append(
                            result)
                        row_data.append(result)

            # Append the collected row data to the main data list for Excel
            if row_data:
                data_for_excel.append(row_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(results)
    # Saving the DataFrame to an Excel file
    df.to_json('output.json')
    print(results)
    return results


base_path = 'finalOutput'
final_results = process_files(base_path)

# ADD FLAGS: If just word detected in col 4 then push flag
