import json
import re
import os


def process_col4(col4):
    if len(col4) == 1 and col4[0] == "0":
        return col4

    processed = []
    current_label = None
    current_sum = 0

    for item in col4:
        if re.search(r'[(){}[\]]', item):
            continue

        if item.startswith("#") or not item.isdigit():
            if current_label is not None:
                processed.append(
                    {'name': current_label,  'value': current_sum})
                current_sum = 0  # Reset the sum for the new label
            current_label = item
        elif item.isdigit():  # Add digit to current sum
            current_sum += int(item)

    if current_label is not None:
        processed.append({'name': current_label,  'value': current_sum})

    return processed


def process_json(data):
    for page_number, page_content in data.items():
        for row_number, row_content in page_content.items():
            col4 = row_content.get('4', [])
            processed_col4 = process_col4(col4)
            row_content['4'] = processed_col4

    return data


def should_merge(current_row, next_row):
    current_row_1 = current_row.get('1', [])
    current_row_2 = current_row.get('2', [])
    current_row_3 = current_row.get('3', [])
    current_row_4 = current_row.get('4', [])

    next_row_1 = next_row.get('1', [])
    next_row_2 = next_row.get('2', [])
    next_row_3 = next_row.get('3', [])
    next_row_4 = next_row.get('4', [])

    current_conditions = (
        len(current_row_1) > 0 and current_row_1[0].isdigit(),
        len(current_row_2) > 0 and (current_row_2[0].isalpha(
        ) or current_row_2[0].replace(' ', '').isalnum()),
        len(current_row_3) > 0 and current_row_3[0] == 'N/A',
        len(current_row_4) > 0 and current_row_4[0] == 'N/A'
    )

    next_conditions = (
        len(next_row_1) > 0 and next_row_1[0] == 'N/A',
        len(next_row_2) > 0 and (next_row_2[0].isalpha(
        ) or next_row_2[0].replace(' ', '').isalnum()),
        len(next_row_3) > 0 and next_row_3[0] != 'N/A',
        len(next_row_4) > 0 and next_row_4[0] != 'N/A'
    )

    print(f"Current row conditions: {current_conditions}")
    print(f"Next row conditions: {next_conditions}")

    return all(current_conditions) and all(next_conditions)


def merge_rows(current_row, next_row):
    current_row['3'] = next_row['3']
    current_row['4'] = next_row['4']
    print(f"Merged rows: {current_row}")


def special_case(data):
    sorted_pages = sorted(data.keys(), key=int)
    for index, page_number in enumerate(sorted_pages):
        page_content = data[page_number]
        page_keys = list(page_content.keys())
        i = 0
        while i < len(page_keys):
            row_number = page_keys[i]
            row_content = page_content[row_number]

            next_row_content = None
            next_row_number = None

            if i == len(page_keys) - 1 and index < len(sorted_pages) - 1:
                next_page_number = sorted_pages[index + 1]
                next_page_content = data[next_page_number]
                next_page_keys = list(next_page_content.keys())
                if next_page_keys:
                    next_row_number = next_page_keys[0]
                    next_row_content = next_page_content[next_row_number]
            elif i < len(page_keys) - 1:
                next_row_number = page_keys[i + 1]
                next_row_content = page_content[next_row_number]

            if next_row_content and should_merge(row_content, next_row_content):
                print(
                    f"Merging Page {page_number} Row {row_number} with Page {next_page_number if next_row_number in next_page_keys else page_number} Row {next_row_number}")
                merge_rows(row_content, next_row_content)
                if next_row_number in next_page_keys:
                    del data[next_page_number][next_row_number]
                else:
                    del page_content[next_row_number]
                    # Update page keys since the row is removed
                    page_keys.remove(next_row_number)
            else:
                print(f"No merge for Page {page_number} Row {row_number}")

            i += 1

    return data


def validate_json(data):
    for page_number, page_content in data.items():
        previous_valid_col3_value = None

        for row_number, row_content in list(page_content.items()):
            if (row_content.get('1', []) == ['N/A'] and
                row_content.get('3', []) == ['N/A'] and
                    row_content.get('4', []) == ['N/A']):
                del page_content[row_number]
                print(
                    f"Page {page_number}, Row {row_number} removed due to specific structure (col1, col3, col4 all 'N/A')")
                continue
            if all(value == ['N/A'] for value in row_content.values()):
                del page_content[row_number]
                print(
                    f"Page {page_number}, Row {row_number} removed: All columns are 'N/A'")
                continue  # Skip further processing for this row

            col3 = row_content.get('3', [])
            col4 = row_content.get('4', [])

            if col3 == ["N/A"]:
                print(
                    f"Page {page_number}, Row {row_number}, Column 3 has 'N/A', skipping...")
                continue

            numeric_values = [value for value in col3 if value.isdigit()]
            non_numeric_values = [
                value for value in col3 if not value.isdigit()]

            if len(numeric_values) != 1 or non_numeric_values:
                row_content['3'] = [
                    f"{col3[0]}_flag"] if numeric_values else [f"number_flag"]
                print(
                    f"Page {page_number}, Row {row_number}, Column 3 flagged due to invalid data")

            elif numeric_values:
                current_col3_value = int(numeric_values[0])
                col4_numeric_sum = sum(int(value)
                                       for value in col4 if value.isdigit())

                if previous_valid_col3_value is not None:
                    expected_value = previous_valid_col3_value - col4_numeric_sum
                    if expected_value != current_col3_value:
                        row_content['3'] = [f"{current_col3_value}_flag"]
                        print(
                            f"Page {page_number}, Row {row_number}, Column 3 flagged due to calculation mismatch")
                    else:
                        print(
                            f"Page {page_number}, Row {row_number}, Column 3 value is correct as expected.")
                else:
                    print(
                        f"Page {page_number}, Row {row_number}, Column 3 skipped: No previous value for validation")

                previous_valid_col3_value = current_col3_value

    return data


def validate_col3_reCheck(file_path):
    # Load the data from the specified file path
    with open(file_path, 'r') as file:
        data = json.load(file)

    previous_valid_col3_value = None

    for page_number, page_content in data.items():
        for row_number, row_content in page_content.items():
            col3 = row_content.get('3', [])
            col4 = row_content.get('4', [])

            if col3 == ["N/A"]:
                continue  # Skip if col3 is explicitly marked as "N/A"

            # Extract numbers from col3, handling both flagged and unflagged numbers
            numeric_values = []
            flags_detected = False  # Flag to track if any flags were found
            for value in col3:
                if '_flag' in value:
                    base_part = value.split('_flag')[0]
                    flags_detected = True
                else:
                    base_part = value

                if base_part.isdigit():
                    numeric_values.append(int(base_part))

            # Calculate the sum of numeric values in col4
            col4_numeric_sum = 0
            for item in col4:
                if isinstance(item, dict):
                    col4_numeric_sum += sum(int(v)
                                            for v in item.values() if isinstance(v, int))
                elif isinstance(item, int):
                    col4_numeric_sum += item

            # Validate and potentially update col3
            if len(numeric_values) != 1:
                row_content['3'] = [
                    f"{col3[0]}_flag"] if numeric_values else ["number_flag"]
                print(
                    f"Page {page_number}, Row {row_number} flagged due to invalid col3 data")
            else:
                current_col3_value = numeric_values[0]
                if previous_valid_col3_value is not None:
                    expected_value = previous_valid_col3_value - col4_numeric_sum
                    if expected_value != current_col3_value:
                        row_content['3'] = [f"{current_col3_value}_flag"]
                        print(
                            f"Page {page_number}, Row {row_number} flagged due to col3 calculation mismatch")
                    else:
                        # Update col3 to just the number if no mismatch and originally flagged
                        if flags_detected:
                            row_content['3'] = [f"{current_col3_value}"]
                        print(
                            f"Page {page_number}, Row {row_number} col3 value is correct as expected.")
                else:
                    print(
                        f"Page {page_number}, Row {row_number} first col3 value, no previous value for validation")

                # Update the previous valid col3 value for the next row
                previous_valid_col3_value = current_col3_value

    # Save the modified data back to the same file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return data


def finalVal(data, idFile):
    idFile = os.path.basename(idFile)
    with open(data, 'r') as file:
        dataF = json.load(file)

    validated_data = validate_json(dataF)
    validated_data = special_case(dataF)
    validated_data = process_json(dataF)
    with open(f'processing/{idFile}.json', 'w') as file:
        json.dump(validated_data, file, indent=4, ensure_ascii=False)


# finalVal(
#     'output_3be7cf7a-1abc-475d-992b-e0c6e5a71927.pdf.json', '3be7cf7a-1abc-475d-992b-e0c6e5a71927.pdf')
