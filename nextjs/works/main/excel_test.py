import pandas as pd
import json
from variables import OPRID, COLUMNHEADING, SCRAP_CODE_MAPPING
import excel_ocr
import os

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def process_data(data, excel_path):
    all_rows = []
    previous_row = None

    for page, contents in data.items():
        for row_id, columns in contents.items():
            operation = columns.get('1', ['N/A'])[0]
            work_center = columns.get('2', ['N/A'])[0]
            op_good_qty = columns.get('3', ['N/A'])[0]
            scrap_details = columns.get('4', [])

            # Filter out invalid rows
            if operation == 'N/A' and work_center == 'N/A' and op_good_qty == 'N/A' and (not scrap_details or scrap_details == ['N/A']):
                continue

            # Convert op_good_qty to integer if it is numeric
            if op_good_qty.isdigit():
                op_good_qty = int(op_good_qty)
            elif op_good_qty == 'N/A':  # Include rows where Op. Good Qty is 'N/A' but still process the scrap details
                op_good_qty = None

            # Initialize dictionary to manage scrap descriptions and quantities
            scrap_quantities = {}
            if scrap_details and isinstance(scrap_details, list) and scrap_details != ['N/A', '0']:
                for item in scrap_details:
                    if isinstance(item, dict) and 'name' in item and 'value' in item:
                        name = item['name']
                        value = item['value']
                        if name == 'о'and value == 0:  # Special handling for entries with the name 'о'
                            name = ''
                        if name not in ["ว", "Samples", "Tensile Test"]:  # Skip excluded descriptions
                            if name in scrap_quantities:
                                scrap_quantities[name] += value
                            else:
                                scrap_quantities[name] = value

            # Check conditions to apply operation and work center from previous valid entry
            if previous_row and operation == 'N/A' and (op_good_qty is not None or scrap_quantities):
                operation = previous_row['operation']
                work_center = previous_row['work_center']

            # Append rows
            if scrap_quantities:
                for description, quantity in scrap_quantities.items():
                # Update the Scrap Code based on the description from the mapping
                    scrap_code = SCRAP_CODE_MAPPING.get(description, '')
                    all_rows.append([work_center, operation, scrap_code, description, op_good_qty if op_good_qty is not None else 0, quantity] + [None] * (len(COLUMNHEADING) - 6))
            else:
                all_rows.append([work_center, operation, None, '', op_good_qty if op_good_qty is not None else 0, 0] + [None] * (len(COLUMNHEADING) - 6))

            # Update previous_row for next iteration
            previous_row = {
                'operation': operation,
                'work_center': work_center,
                'op_good_qty': op_good_qty,
                'scrap_quantities': scrap_quantities
            }

    df = pd.DataFrame(all_rows, columns=COLUMNHEADING)

    # Synchronize the columns with the same data but different headers
    df['Parent Good qty'] = df['Entered Good Qty'] = df['Op. Good Qty']
    df['Parent Scrap qty'] = df['Entered Scrap Qty'] = df['Op. Scrap Qty']

    df.to_excel(excel_path, index = False)
    print(f"Excel file has been updated at {excel_path}")
    return df, len(all_rows)

'''
def save_to_excel(df, filename):
    df.to_excel(filename, index=False)
    print(f"Excel file has been saved to {filename}")
'''

def main(firstPage_imagePath, restPages_jsonPath):
    # Define the path to the output Excel file
    excel_output_file = os.path.join(os.getcwd(), 'final_output.xlsx')
    
    # Initialize or load the DataFrame
    if os.path.exists(excel_output_file):
        df = pd.read_excel(excel_output_file)
    else:
        df = pd.DataFrame(columns=COLUMNHEADING)

    # Load JSON data
    json_data = load_json(restPages_jsonPath)
    # Process data and get updated DataFrame
    df, json_row_count = process_data(json_data, excel_output_file)
    
    # Save the updated DataFrame to Excel
    df.to_excel(excel_output_file, index=False)
    print(f"JSON data updated to Excel at {excel_output_file}")
    
    # Process OCR data and update the Excel file
    excel_ocr.process_image_for_ocr(firstPage_imagePath, excel_output_file, json_row_count)

    
# Testing
firstPage_imagePath = '/Users/zyy/Desktop/TE/excel_update/first_page_dataset/7855.png'
restPages_jsonPath = '/Users/zyy/Documents/GitHub/TE-AI-Cup/nextjs/works/main/test.json'

main(firstPage_imagePath, restPages_jsonPath)
