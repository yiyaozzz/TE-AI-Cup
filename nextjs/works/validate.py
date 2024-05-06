import json
import sys
import re


def validate_col3_with_flags(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    for page_number, page_content in data.items():
        previous_valid_col3_value = None

        for row_number, row_content in page_content.items():
            if (row_content.get('1', []) == ['N/A'] and
                row_content.get('3', []) == ['N/A'] and
                    row_content.get('4', []) == ['N/A']):
                del page_content[row_number]
                # print(
                #     f"Page {page_number}, Row {row_number} removed due to specific structure (col1, col3, col4 all 'N/A')")
                continue

            if all(value == ['N/A'] for value in row_content.values()):
                del page_content[row_number]
                # print(
                #     f"Page {page_number}, Row {row_number} removed: All columns are 'N/A'")
                continue

            col3 = row_content.get('3', [])
            col4 = row_content.get('4', [])

            if col3 == ["N/A"]:
                # print(
                #     f"Page {page_number}, Row {row_number}, Column 3 has 'N/A', skipping...")
                continue

            numeric_values = []
            flags_detected = False
            for value in col3:
                if '_flag' in value:
                    numeric_part = value.split('_flag')[0]
                    flags_detected = True
                    if numeric_part.isdigit():
                        numeric_values.append(int(numeric_part))
                elif value.isdigit():
                    numeric_values.append(int(value))

            col4_numeric_sum = 0
            for item in col4:
                if isinstance(item, dict):
                    if '_flag' not in item['name']:
                        col4_numeric_sum += int(item['value'])

            if len(numeric_values) != 1:
                row_content['3'] = [
                    f"{col3[0]}_flag"] if numeric_values else ["number_flag"]
                # print(
                #     f"Page {page_number}, Row {row_number}, Column 3 flagged due to invalid data")
            else:
                current_col3_value = numeric_values[0]
                if previous_valid_col3_value is not None:
                    expected_value = previous_valid_col3_value - col4_numeric_sum
                    if expected_value != current_col3_value:
                        row_content['3'] = [f"{current_col3_value}_flag"]
                        # print(
                        #     f"Page {page_number}, Row {row_number}, Column 3 flagged due to calculation mismatch")
                    else:
                        if flags_detected:
                            row_content['3'] = [str(current_col3_value)]
                        # print(
                        #     f"Page {page_number}, Row {row_number}, Column 3 value is correct as expected.")

                previous_valid_col3_value = current_col3_value
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return data


def search_first_flag(data, path=''):
    for page, page_content in data.items():
        for row, row_content in page_content.items():
            for col, values in row_content.items():
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and '_flag' in value:
                            return f"{page}/{row}/{col}/{value}"
                        elif isinstance(value, dict) and '_flag' in value.get('name', ''):
                            return f"{page}/{row}/{col}/{value['name']}"
                elif isinstance(values, dict):
                    if '_flag' in values.get('name', ''):
                        return f"{page}/{row}/{col}/{value['name']}"

    return None


def main():
    pdf_path = sys.argv[1]
    with open(pdf_path, 'r') as file:
        data = json.load(file)

    validate_col3_with_flags(pdf_path)
    result = search_first_flag(data)

    if result:
        flag_location = result
        parts = flag_location.split('/')
        pageNum = parts[0]
        flag_location = '/'.join(parts[:-1])
        print(f"{flag_location}")
        if int(pageNum) == 1:
            print(pageNum)
        elif int(pageNum) == 5:
            print("1")
        else:
            print(int(pageNum)-1)


if __name__ == "__main__":
    main()
