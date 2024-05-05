import pandas as pd
import json

from variables import OPRID, COLUMNHEADING

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

            # Filter out invalid row
            if operation == 'N/A' and work_center == 'N/A' and op_good_qty == 'N/A' and (not scrap_details or scrap_details[0] == 'N/A'):
                continue
            if operation == 'N/A' and op_good_qty == 'N/A' and (not scrap_details or scrap_details[0] == 'N/A'):
                continue         

            # Convert op_good_qty to integer if it is numeric
            if op_good_qty.isdigit():
                op_good_qty = int(op_good_qty)
            elif op_good_qty == 'N/A':  # Include rows where Op. Good Qty is 'N/A' but still process the scrap details
                op_good_qty = None

            # Handle empty or placeholder scrap details explicitly
            if not scrap_details or scrap_details[0] in ['N/A', '0']:
                all_rows.append([work_center, operation, None, '', 0 if op_good_qty is None else op_good_qty, 0] + [None] * (len(COLUMNHEADING) - 6))
                continue

                    
            # Initialize dictionary to manage scrap descriptions and quantities
            scrap_quantities = {}
            current_description = None


            # Applying logic to pass operation and work center from previous valid entry
            if previous_row and previous_row['op_good_qty'] is None and not previous_row['scrap_quantities'] and operation == 'N/A':
                operation = previous_row['operation']
                work_center = previous_row['work_center']
                    
            # Handle Col4
            for item in scrap_details:
                if item in ['N/A', '0']:
                    # If 'N/A' or '0', append an empty description with zero quantity
                    all_rows.append([work_center, operation, None, '', op_good_qty if op_good_qty != 'N/A' else 0, 0] + [None] * (len(COLUMNHEADING) - 6))
                    continue  # Skip further processing for this row
                elif isinstance(item, str) and not item.isdigit() and item not in ["ว", "Samples", "Tensile Test"]:
                    # Handle valid descriptions
                    current_description = OPRID.get(item, item)
                    scrap_quantities[current_description] = 0
                elif item.isdigit() and current_description:
                    # Add quantities to the current description
                    scrap_quantities[current_description] += int(item)


            
            # Process based on consecutive row logic
            if previous_row:
                if previous_row['op_good_qty'] == op_good_qty:
                    # Check scrap quantity adjustment for consecutive rows with the same Op. Good Qty
                    if scrap_details == ['0']:  # Only update if scrap is '0'
                        scrap_quantities = {desc: 0 for desc in scrap_quantities}
                    else:
                        print(f"_flag: Non-zero scrap quantity on consecutive rows with the same Op. Good Qty at Page {page}, Rows {previous_row['row_id']} and {row_id}")
                        print(f"Previous row data: {previous_row}")
                        print(f"Current row data: Operation: {operation}, Work Center: {work_center}, Op. Good Qty: {op_good_qty}, Scrap: {scrap_quantities}")
                        # Force zero scrap quantity
                        scrap_quantities = {desc: 0 for desc in scrap_quantities}         
                else:
                    # Check for expected Op. Good Qty
                    expected_qty = previous_row['op_good_qty'] - sum(scrap_quantities.values())
                    if op_good_qty != expected_qty:
                        print(f"_flag: Incorrect Op. Good Qty at Page {page}, Rows {previous_row['row_id']} and {row_id}")
                        print(f"Previous row data: {previous_row}")
                        print(f"Current row data: Operation: {operation}, Work Center: {work_center}, Op. Good Qty: {op_good_qty}, Scrap: {scrap_quantities}")
                        op_good_qty = expected_qty  # Correct the Op. Good Qty

            # Update previous_row for the next iteration
            previous_row = {
                'row_id': row_id,
                'operation': operation,
                'work_center': work_center,
                'op_good_qty': op_good_qty,
                'scrap_quantities': scrap_quantities,

            }

            # Append rows
            for description, quantity in scrap_quantities.items():
                row = [work_center, operation, None, description, op_good_qty, quantity] + [None] * (len(COLUMNHEADING) - 6)
                all_rows.append(row)

    return pd.DataFrame(all_rows, columns=COLUMNHEADING)


def save_to_excel(df, filename):
    df.to_excel(filename, index=False)
    print(f"Excel file has been saved to {filename}")

# Example usage:
json_file = 'output_test.json'
excel_output_file = 'updated_output_test.xlsx'

# Load data
json_data = load_json(json_file)

# Process data
processed_data = process_data(json_data)

# Save to Excel
save_to_excel(processed_data, excel_output_file)
