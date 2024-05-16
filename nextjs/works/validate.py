import json
import sys
import re
from finalSheet import excel_prod
import os
# from main.excelSheet import sheetmain


def validate_col3_with_flags(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    previous_col3_value = None
    for page, rows in data.items():
        for row, columns in rows.items():
            col4_items = columns.get('4', [])
            sum_col4 = sum(safe_int(item.get('value', 0))
                           for item in col4_items if isinstance(item, dict))

            current_col3 = columns.get('3', [])
            current_col3_value = safe_convert_to_int(
                current_col3[0]) if current_col3 else None

            handle_flags(page, row, columns, previous_col3_value,
                         current_col3_value, sum_col4)

            # Update previous_col3_value for the next iteration
            previous_col3_value = safe_convert_to_int(
                current_col3[0]) if current_col3 and current_col3[0] != "N/A" else None

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return data


def handle_flags(page, row, columns, previous_col3_value, current_col3_value, sum_col4):
    if page == '1':
        if columns.get('3', [''])[0] == "N/A":
            return
        elif current_col3_value != 500:
            columns['3'] = [f"{columns['3'][0]}_flag"]
    else:
        if previous_col3_value is not None and current_col3_value is not None:
            expected_col3_value = previous_col3_value - sum_col4
            if expected_col3_value != current_col3_value:
                columns['3'] = [f"{columns['3'][0]}_flag"]


def safe_convert_to_int(value):
    """Convert values to integer or return None if not possible"""
    try:
        return int(value.replace('_flag', '')) if '_flag' in value else int(value)
    except ValueError:
        return None


def safe_int(value):
    """Safely convert value to integer, returning 0 if conversion fails"""
    try:
        return int(value)
    except ValueError:
        return 0


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

    validate_col3_with_flags(pdf_path)
    with open(pdf_path, 'r') as file:
        data = json.load(file)

    uidVal = os.path.basename(pdf_path)
    # print('UID VAL PATH: '+uidVal)
    uidVal = uidVal.split('.')[0]
    uidVal = uidVal.split('output_')[-1]
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

        if int(pageNum) == 1:
            print("5")
        elif int(pageNum) == 17:
            print("17")
        else:
            print(int(pageNum) + 1)
    # #     # UID.exlsx
    # else:
    #     sheetmain(
    #         f'nextjs/works/uploads/{uidVal}/page_1.png', f'processing/{uidVal}')


if __name__ == "__main__":
    main()
