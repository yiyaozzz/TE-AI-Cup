import os
import pandas as pd
from fuzzywuzzy import process
from api import apiResult
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
        page_path = os.path.join(base_path, page_folder)
        results[page_number] = results.get(page_number, {})

        row_folders = [f for f in os.listdir(
            page_path) if f.startswith("row_")]
        row_folders_sorted = sorted(
            row_folders, key=lambda x: int(x.split('_')[1]))

        for row_folder in row_folders_sorted:
            row_number = row_folder.split('_')[1]
            row_path = os.path.join(page_path, row_folder)
            results[page_number][row_number] = results[page_number].get(
                row_number, {})
            row_data = []  # Initialize list to store row data for Excel

            col_folders = sorted(os.listdir(row_path),
                                 key=lambda x: int(x.split('_')[1]))
            for col_folder in col_folders:
                col_number = col_folder.split('_')[1]
                col_path = os.path.join(row_path, col_folder)
                results[page_number][row_number][col_number] = []

                for file_name in os.listdir(col_path):
                    file_path = os.path.join(col_path, file_name)
                    result = None
                    if col_folder in ['column_1', 'column_2', 'column_3']:
                        if 'Words' in file_name or col_folder != 'column_4':
                            result = apiResult(file_path)
                    elif col_folder == 'column_4':
                        if 'N_A' in file_name:
                            result = 'N/A'
                        elif 'Words-and-tallys' in file_name:
                            result = resnetPred(file_path)
                        elif 'Circled-number' in file_name or 'Number' in file_name:
                            result = apiResult(file_path)

                    if result:
                        results[page_number][row_number][col_number].append(
                            result)
                        row_data.append(result)

            # Append the collected row data to the main data list for Excel
            if row_data:
                data_for_excel.append(row_data)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data_for_excel)
    # Saving the DataFrame to an Excel file
    df.to_excel('output.xlsx', index=False)

    return results

# [[[col1],[col2],[col3],[ew,6,disc,10]],[]]
# Maybe construct own data struct (tree struct?)

# base_path = 'finalOutput1'
# final_results = process_files(base_path)
# print(final_results)
