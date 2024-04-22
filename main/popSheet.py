import fuzzywuzzy
from variables import COLUMNHEADING, OPRID
import os
from api import apiResult
from resnt_test import resnetPred

# def preprocessCol3():


def process_files(base_path):
    """
    The function `process_files` processes files in a directory structure and organizes the results by
    rows and columns.

    :param base_path: The `base_path` parameter in the `process_files` function is the path to the
    directory where the files are located. This function processes files in a specific directory
    structure, extracting information based on the structure of the folders and files within the
    specified base path
    :return: The function `process_files` is returning a list of results, organized by rows and columns.
    The results are processed based on the column type, with different actions taken for each type of
    column. The final results are stored in a nested list structure, where each element corresponds to a
    specific row and column in the input directory structure.
    """
    results = []  # This list will hold all the results, organized by rows and columns

    # Traverse the directory structure
    for page_folder in sorted(os.listdir(base_path)):
        if page_folder.startswith("page_"):
            page_path = os.path.join(base_path, page_folder)

            for row_folder in sorted(os.listdir(page_path)):
                if row_folder.startswith("row_"):
                    row_index = int(row_folder.split('_')[1]) - 1
                    row_path = os.path.join(page_path, row_folder)

                    # Ensure the list has enough sub-lists for each row
                    while len(results) <= row_index:
                        results.append([])

                    for col_folder in sorted(os.listdir(row_path)):
                        col_index = int(col_folder.split('_')[1]) - 1
                        col_path = os.path.join(row_path, col_folder)

                        # Ensure the list for this row has enough sub-lists for each column
                        while len(results[row_index]) <= col_index:
                            results[row_index].append([])

                        # Process files based on the column type
                        if col_folder in ['column_1', 'column_2']:
                            for file_name in os.listdir(col_path):
                                file_path = os.path.join(col_path, file_name)
                                results[row_index][col_index].append(
                                    apiResult(file_path))
                        elif col_folder == 'column_3':
                            first_file = os.listdir(
                                col_path)[0] if os.listdir(col_path) else None
                            if first_file:
                                file_path = os.path.join(col_path, first_file)
                                results[row_index][col_index].append(
                                    apiResult(file_path))
                        elif col_folder == 'column_4':
                            for file_name in os.listdir(col_path):
                                file_path = os.path.join(col_path, file_name)
                                if file_name == 'N_A':
                                    results[row_index][col_index].append('N/A')
                                elif file_name == 'words-and-tallys':
                                    results[row_index][col_index].append(
                                        resnetPred(file_path))
                                elif file_name in ['circled-number', 'number']:
                                    results[row_index][col_index].append(
                                        apiResult(file_path))
    print(results)

    return results


# Example usage
base_path = 'finalOutput'
final_results = process_files(base_path)
print(final_results)
