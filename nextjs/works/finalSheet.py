import pandas as pd
import json
import os
from main.variables import OPRID, COLUMNHEADING


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def process_data(data):
    all_rows = []
    previous_row = None

    for page, contents in data.items():
        for row_id, columns in contents.items():
            operation = columns.get('1', ['N/A'])[0]
            work_center = columns.get('2', ['N/A'])[0]
            op_good_qty = columns.get('3', ['N/A'])[0]
            scrap_details = columns.get('4', [])

            if operation == 'N/A' and work_center == 'N/A' and op_good_qty == 'N/A' and (not scrap_details or scrap_details[0] == 'N/A'):
                continue
            if operation == 'N/A' and op_good_qty == 'N/A' and (not scrap_details or scrap_details[0] == 'N/A'):
                continue

            if op_good_qty.isdigit():
                op_good_qty = int(op_good_qty)
            elif op_good_qty == 'N/A':  # Include rows where Op. Good Qty is 'N/A' but still process the scrap details
                op_good_qty = None

            if not scrap_details or scrap_details[0] in ['N/A', '0']:
                all_rows.append([work_center, operation, None, '',
                                0 if op_good_qty is None else op_good_qty, 0] + [None] * (len(COLUMNHEADING) - 6))
                continue

            scrap_quantities = {}
            current_description = None

            if previous_row and previous_row['op_good_qty'] is None and not previous_row['scrap_quantities'] and operation == 'N/A':
                operation = previous_row['operation']
                work_center = previous_row['work_center']
                previous_row = None
            # Handle Col4
            for item in scrap_details:
                if item in ['N/A', '0']:
                    all_rows.append([work_center, operation, None, '', op_good_qty if op_good_qty !=
                                    'N/A' else 0, 0] + [None] * (len(COLUMNHEADING) - 6))
                    continue  # Skip further processing for this row
                elif isinstance(item, str) and not item.isdigit() and item not in ["ว", "Samples", "Tensile Test"]:
                    current_description = OPRID.get(item, item)
                    scrap_quantities[current_description] = 0
                elif item.isdigit() and current_description:
                    scrap_quantities[current_description] += int(item)

            if previous_row:
                if previous_row['op_good_qty'] == op_good_qty:
                    if scrap_details == ['0']:
                        scrap_quantities = {
                            desc: 0 for desc in scrap_quantities}
                    else:
                        print(
                            f"_flag: Non-zero scrap quantity on consecutive rows with the same Op. Good Qty at Page {page}, Rows {previous_row['row_id']} and {row_id}")
                        print(f"Previous row data: {previous_row}")
                        print(
                            f"Current row data: Operation: {operation}, Work Center: {work_center}, Op. Good Qty: {op_good_qty}, Scrap: {scrap_quantities}")
                        scrap_quantities = {
                            desc: 0 for desc in scrap_quantities}
                else:

                    expected_qty = previous_row['op_good_qty'] - \
                        sum(scrap_quantities.values())
                    if op_good_qty != expected_qty:
                        print(
                            f"_flag: Incorrect Op. Good Qty at Page {page}, Rows {previous_row['row_id']} and {row_id}")
                        print(f"Previous row data: {previous_row}")
                        print(
                            f"Current row data: Operation: {operation}, Work Center: {work_center}, Op. Good Qty: {op_good_qty}, Scrap: {scrap_quantities}")
                        op_good_qty = expected_qty

            previous_row = {
                'row_id': row_id,
                'operation': operation,
                'work_center': work_center,
                'op_good_qty': op_good_qty,
                'scrap_quantities': scrap_quantities,

            }

            for description, quantity in scrap_quantities.items():
                row = [work_center, operation, None, description,
                       op_good_qty, quantity] + [None] * (len(COLUMNHEADING) - 6)
                all_rows.append(row)

    return pd.DataFrame(all_rows, columns=COLUMNHEADING)


def save_to_excel(df, filename):
    os.makedirs('finalSheets', exist_ok=False)
    path = os.path.basename(filename)
    print("PATH " + path)
    df.to_excel(filename, index=False)
    print(f"Excel file has been saved to {filename}")


def excel_prod(json_file, excel_output_file):

    json_data = load_json(json_file)

    processed_data = process_data(json_data)

    save_to_excel(processed_data, excel_output_file)


# json_file = 'nextjs/works/output_b828b7a5-50e5-4f16-92f3-511703cdb07f.pdf.json'
# excel_output_file = 'updated_output_test.xlsx'
# excel_prod(json_file, excel_output_file)
