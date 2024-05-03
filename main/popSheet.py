import os
import pandas as pd
from fuzzywuzzy import process
from gapi import apiResult
from tallyYolo import predict_and_show_labels
from resnt_test import resnetPred
import json
from variables import OPRID
from api import aapiResult


def get_closest_match(word, dictionary=OPRID, threshold=70):
    closest_match = process.extractOne(word, dictionary.keys())
    if closest_match and closest_match[1] >= threshold:
        # closest_match returns a tuple (matching_key, score), check if score meets the threshold
        matching_key = closest_match[0]

        return dictionary[matching_key]
    else:
        return word


def process_files(base_path):
    results = {}  # Dictionary to hold results organized by pages, rows, and columns
    data_for_excel = []  # List to hold data for Excel export

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
            row_data = []

            col_folders = sorted(os.listdir(row_path),
                                 key=lambda x: int(x.split('_')[1]))

            for col_folder in col_folders:
                col_number = col_folder.split('_')[1]
                col_path = os.path.join(row_path, col_folder)
                sort_file = sorted(os.listdir(col_path),
                                   key=lambda x: int(x.split('_')[0]))

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
                            if result == 'Number_1':
                                result = '1'
                        elif 'Words' in file_name:
                            result = apiResult(file_path)
                            if result.isalpha() == False:
                                print("VALUE NOT ENG ", result)
                                result = aapiResult(file_path)
                            if result == "태":
                                result = "EH"
                            elif result == "나":
                                result == "LT"
                            elif result == "EN":
                                result = "EW"
                            if result is not None or result is not 'None':
                                result = get_closest_match(result)

                        elif 'Circled_Number' in file_name:
                            continue
                        elif 'Number' in file_name:
                            result = apiResult(file_path)
                            if result is None or result == 'None':
                                result = '0'
                            elif result.isnumeric() == False:
                                result = aapiResult(file_path)
                                if result is None or result == 'None':
                                    result = '0'
                            elif result.upper() == 'N/A':
                                result = "N/A"
                    if result:
                        results[page_number][row_number][col_number].append(
                            result)
                        row_data.append(result)

            if row_data:
                data_for_excel.append(row_data)

    with open('output.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(results)
    return results


# base_path = 'finalOutput'
# final_results = process_files(base_path)

# ADD FLAGS: If just word detected in col 4 then push flag
