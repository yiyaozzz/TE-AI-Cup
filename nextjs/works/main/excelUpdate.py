import pandas as pd
import json

# Load the JSON data
with open('/Users/zyy/Documents/GitHub/TE-AI-Cup/output.json') as file:
    data = json.load(file)

# Define the header for the DataFrame to be used in Excel
header = ["Work Center", "Operation", "Scrap Code", "Scrap Description",
          "Op. Good Qty", "Op. Scrap Qty", "UoM", "PPM__________",
          "Posting date", "Entry Date", "Prod Order", "Material Number",
          "Material Description", "Parent Good qty", "Parent Scrap qty",
          "Order Unit", "Order Type", "Plant", "Entered Good Qty",
          "Entered Scrap Qty", "Entered UoM"]

# Process each page in the JSON data
all_rows = []
for page, content in data.items():
    for row_key, row_val in content.items():
        if row_val is None:  # Skip null rows
            continue
        operation = row_val.get('1', [''])[0] if row_val.get('1') else ''
        work_center = row_val.get('2', [''])[0] if row_val.get('2') else ''
        op_good_qty = row_val.get('3', [''])[0] if row_val.get('3') else ''
        scrap_details = row_val.get('4', [])
        # Handle empty or N/A column 4
        if not scrap_details or scrap_details == ['N/A']:
            # Leave "Scrap Description" and "Op. Scrap Qty" as empty
            scrap_details = ['', '0']

        # Initialize dictionary to store scrap quantities
        scrap_quantities = {}
        current_description = None
        for detail in scrap_details:
            # Check for non-numeric
            if isinstance(detail, str) and (detail.isalpha() or not detail.isdigit()):
                # Save accumulated quantity for previous description
                if current_description is not None:
                    all_rows.append([work_center, operation, None, current_description, op_good_qty,
                                     scrap_quantities.get(current_description, ''), None] + [None]*(len(header)-7))
                current_description = detail
                # Initialize or reset for new description
                scrap_quantities[current_description] = 0
            elif detail.isdigit():  # Accumulate quantities
                if current_description:
                    scrap_quantities[current_description] += int(detail)

        # Append the last accumulated item if it exists
        if current_description is not None:
            all_rows.append([work_center, operation, None, current_description, op_good_qty,
                             scrap_quantities.get(current_description, ''), None] + [None]*(len(header)-7))
        # If no details, append blank scrap description
        elif current_description is None and (not scrap_details or scrap_details == ['']):
            all_rows.append([work_center, operation, None, '', op_good_qty, '',
                             None] + [None]*(len(header)-7))

# Create DataFrame from the collected data
df = pd.DataFrame(all_rows, columns=header)

# Save the DataFrame to a new Excel file
# df.to_excel('/Users/zyy/Documents/GitHub/TE-AI-Cup/updated_output.xlsx', index=False)

# print("A new Excel file has been created with the updated data.")
