import os
import pandas as pd
from fuzzywuzzy import process
from main.gapi import apiResult
from main.tallyYolo import predict_and_show_labels
import json
from nextjs.works.main.variables import OPRID
from main.api import aapiResult
from dimVal import dimValPred


# def get_closest_match(word, dictionary=OPRID, threshold=70, image=''):
#     # if any(char.isdigit() for char in word):
#     #     word = dimValPred(image)
#     #     if word is None:
#     #         return 'wordNotFound_flag'
#     if word is None:
#         word = aapiResult(image)
#         if word is None:
#             return 'wordNotFound_flag'

#     if not isinstance(word, str):
#         word = str(word).upper()
#     closest_match = process.extractOne(word, dictionary.keys())

#     if closest_match and closest_match[1] >= threshold:
#         matching_key = closest_match[0]

#         return dictionary[matching_key]
#     else:
#         result = aapiResult(image)

#         if result is not None or result != 'None':
#             result = str(result).upper()
#             closest_match2 = process.extractOne(result, dictionary.keys())
#             if closest_match2 and closest_match2[1] >= threshold:
#                 matching_key = closest_match2[0]

#                 return dictionary[matching_key]
#             else:
#                 return word + '_flag'
#         else:
#             return word + '_flag'

# def get_closest_match(word, dictionary=OPRID, threshold=70, image=''):

#     if word is None:
#         return 'wordNotFound_flag'
#     if not isinstance(word, str):
#         word = str(word).upper()
#     closest_match = process.extractOne(word, dictionary.keys())

#     if closest_match and closest_match[1] >= threshold:
#         matching_key = closest_match[0]

#         return dictionary[matching_key]
#     else:
#         result = aapiResult(image)

#         if result is not None or result != 'None':
#             result = str(result).upper()
#             closest_match2 = process.extractOne(result, dictionary.keys())
#             if closest_match2 and closest_match2[1] >= threshold:
#                 matching_key = closest_match2[0]

#                 return dictionary[matching_key]
#             else:
#                 return word + '_flag'
#         else:
#             return word + '_flag'

def get_closest_match(word, dictionary=OPRID, threshold=80, image=''):
    if word is not None:
        word = word.upper()
        closest_match = process.extractOne(
            word, dictionary.keys(), score_cutoff=threshold)
        if closest_match:
            return dictionary[closest_match[0]]

    result = aapiResult(image)
    if result is None or result == 'None':
        return 'wordNotFound_flag'  # No valid word found from API

    result = str(result).upper()
    # Check if there's any numeric value in the result
    if any(char.isdigit() for char in result):
        dim_pred = dimValPred(image)
        if dim_pred is None:
            return 'wordNotFound_flag'
        # Process the prediction
        dim_pred = process_dim_prediction(dim_pred)
        closest_match3 = process.extractOne(
            dim_pred, dictionary.keys(), score_cutoff=threshold)
        if closest_match3:
            return dictionary[closest_match3[0]]
        return dim_pred + '_flag'

    closest_match2 = process.extractOne(
        result, dictionary.keys(), score_cutoff=threshold)
    if closest_match2:
        return dictionary[closest_match2[0]]

    return result + '_flag'


def process_dim_prediction(pred):
    """ Process the prediction from dimValPred and adjust the string if necessary. """
    pred = str(pred).upper().split()
    # Ensure there are at least 4 items and the second item is '0'
    if len(pred) >= 4 and pred[1] == '0':
        pred[1] = 'O'  # Change '0' to 'O'
    return ' '.join(pred)


def process_files(base_path, uid="ff"):
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
                            if result is not None and result != 'None':
                                if isinstance(result, str) and not result.isnumeric():
                                    result = aapiResult(file_path)
                                    if result is not None and result != 'None':
                                        if isinstance(result, str) and not result.isnumeric():
                                            result = 'ComptQTY_flag'
                        elif 'Word' in file_name:
                            result = apiResult(file_path)
                            if result is not None and result != 'None':
                                if isinstance(result, str) and not result.isnumeric():
                                    result = aapiResult(file_path)
                                    if result is not None and result != 'None':
                                        if isinstance(result, str) and not result.isnumeric():
                                            result = 'ComptQTY_flag'
                    elif col_folder == 'column_4':
                        if 'N_A' in file_name or 'N-A' in file_name:
                            result = 'N/A'
                        elif 'Words-and-tallys' in file_name:
                            result = predict_and_show_labels(file_path)
                            if result == 'Number_1':
                                result = '1'
                        elif 'Words' in file_name:
                            result = apiResult(file_path)
                            if result == "태":
                                result = "EH"
                            elif result == "나":
                                result == "LT"
                            elif result == "EN":
                                result = "EW"

                            result = get_closest_match(result, image=file_path)

                        elif 'Circled_Number' in file_name:
                            continue
                        elif 'Number' in file_name:
                            result = apiResult(file_path)
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

    with open(f'output_{uid}.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(results)
    return results


base_path = 'nextjs/works/finalOutput_b2518793-8e06-4ec7-aa7a-954ee56bb933.pdf'
final_results = process_files(base_path)

# base_path = 'finalOutput_b2518793-8e06-4ec7-aa7a-954ee56bb933.pdf'
# final_results = process_files(base_path)

# ADD FLAGS: If just word detected in col 4 then push flag
