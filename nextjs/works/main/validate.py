import json
import re
import os


def process_json(data):
    for page_number, page_content in data.items():
        for row_number, row_content in page_content.items():
            col4 = row_content.get('4', [])
            processed_col4 = process_col4(col4)
            row_content['4'] = processed_col4

    return data


def process_and_flag_data(json_data):
    previous_col3_value = None  # To keep track of the previous row's column 3 value
    for page, rows in json_data.items():
        for row, columns in rows.items():
            col4_items = columns.get('4', [])
            sum_col4 = sum(item.get('value', 0)
                           for item in col4_items if isinstance(item, dict))
            print(f"Page {page}, Row {row}, Sum of col4 values: {sum_col4}")

            current_col3 = columns.get('3', [])
            current_col3_value = int(
                current_col3[0]) if current_col3 and current_col3[0].isdigit() else None

            if page == '1':
                if current_col3[0] == "N/A":
                    print(
                        f"No action on page 1, row {row}, because col3 is 'N/A'.")
                elif current_col3_value != 500:
                    print(
                        f"Flag raised on page 1, row {row}, because col3 is not 500.")
                    columns['3'] = [f"{current_col3[0]}_flag"]
            elif page == '5':
                if previous_col3_value == "N/A":
                    if current_col3[0].isdigit() and current_col3[0] == 500:
                        print(
                            f"Page 5, row {row}: col3 is numeric as expected when previous value was 'N/A'. No flag raised.")
                    else:
                        columns['3'] = [f"{current_col3[0]}_flag"]
                        print(
                            f"Page 5, row {row}: Warning - col3 is not numeric when previous value was 'N/A'.")
                else:
                    expected_col3_value = int(previous_col3_value) - \
                        int(sum_col4) if previous_col3_value.isdigit() else None
                    if expected_col3_value is not None and expected_col3_value != current_col3_value:
                        print(
                            f"Flag raised on page {page}, row {row}. Expected col3 value: {expected_col3_value}, found: {current_col3_value}")
                        columns['3'] = [f"{current_col3[0]}_flag"]
            else:
                if previous_col3_value is not None and current_col3_value is not None and previous_col3_value is not 'N/A' and current_col3 is not 'N/A':
                    expected_col3_value = int(
                        previous_col3_value) - int(sum_col4)
                    if expected_col3_value != current_col3_value:
                        print(
                            f"Flag raised on page {page}, row {row}. Expected col3 value: {expected_col3_value}, found: {current_col3_value}")
                        columns['3'] = [f"{current_col3[0]}_flag"]

            # Update previous_col3_value for the next iteration
            previous_col3_value = current_col3[0] if current_col3 and current_col3[0] != "N/A" else "N/A"


def process_col4(col4):
    if len(col4) == 1 and col4[0] == "0":
        return col4

    processed = []
    current_label = None
    current_sum = 0
    last_valid_label_index = -1

    for item in col4:
        if re.search(r'[(){}[\]]', item):
            continue

        if item.startswith("#") or not item.isdigit():
            if current_label is not None:
                if current_label in ["Samples", "Tensile Test"] or "TT" in current_label or "SP" in current_label or "(" in current_label or ")" in current_label:
                    if last_valid_label_index != -1:
                        processed[last_valid_label_index]['value'] += current_sum
                else:
                    processed.append(
                        {'name': current_label, 'value': current_sum})
                    last_valid_label_index += 1
                current_sum = 0
            current_label = item
        elif item.isdigit():
            current_sum += int(item)

    if current_label is not None:
        if current_label in ["Samples", "Tensile Test"] or "TT" in current_label or "SP" in current_label or "(" in current_label or ")" in current_label:
            if last_valid_label_index != -1:
                processed[last_valid_label_index]['value'] += current_sum
        else:
            processed.append({'name': current_label, 'value': current_sum})

    return processed


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
                    page_keys.remove(next_row_number)
            else:
                print(f"No merge for Page {page_number} Row {row_number}")

            i += 1

    return data


def clean_data(json_data):
    pages_to_delete = []
    for page, rows in json_data.items():
        rows_to_delete = []
        for row, columns in rows.items():
            if columns.get('1', []) == ["N/A"] and columns.get('3', []) == ["N/A"] and columns.get('4', []) == ["N/A"]:
                rows_to_delete.append(row)
            else:
                col3_items = columns.get('3', [])
                cleaned_col3 = ["N/A"]
                for item in col3_items:
                    if item.isdigit():
                        cleaned_col3 = [item]
                        break
                columns['3'] = cleaned_col3

        for row in rows_to_delete:
            del rows[row]

        if not rows:
            pages_to_delete.append(page)

    for page in pages_to_delete:
        del json_data[page]


def finalVal(data, idFile):
    idFile = os.path.basename(idFile)
    with open(data, 'r') as file:
        dataF = json.load(file)

    clean_data(dataF)
    special_case(dataF)
    process_json(dataF)
    process_and_flag_data(dataF)
    with open(f'processing/{idFile}.json', 'w') as file:
        json.dump(dataF, file, indent=4, ensure_ascii=False)


# finalVal(
#     'output_66695640-9ca2-4705-8d28-5566a3c32270.pdf.json', '3be7cf7a-1abc-475d-992b-e0c6e5a71927.pdf')
