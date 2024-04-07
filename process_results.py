import pandas as pd


def preprocess_rows(rows):
    processed_rows = []
    for i in range(len(rows)):
        # Check if the first item is 'n/a'; skip adding items for this row.
        if rows[i][0].lower() == 'n/a':
            continue

        # If the next row exists and its first cell is 'n/a' but has data in other cells,
        # use that data for the current row if the current row's next three cells are 'n/a'.
        if i+1 < len(rows) and all(item.lower() == 'n/a' for item in rows[i][1:4]) and rows[i+1][0].lower() == 'n/a':
            processed_rows.append([rows[i][0]] + rows[i+1][1:4])
        else:
            processed_rows.append(rows[i])

    return processed_rows


def save_to_excel(processed_rows, columns, file_path):
    df = pd.DataFrame(processed_rows, columns=columns)
    df.to_excel(file_path, index=False)


def process(label_list):
    complete(label_list)
    return None


def complete(rows):
    columns = ["First Item", "Second Item", "Third Item", "Fourth Item"]
    processed_rows = preprocess_rows(rows)
    save_to_excel(processed_rows, columns, "processed_predictions.xlsx")


# <------------- Importing into excel sheet ------------->
COLUMNS = [
    "Work Center", "Operation", "Confirmation Number", "Confirmation Counter",
    "Scrap Code", "Scrap Description", "Short Text", "Op. Good Qty",
    "Op. Scrap Qty", "UoM", "Posting date", "Posted by", "Entry Date",
    "Entry Time", "Prod Order", "Material Number", "Material Description",
    "Parent Good qty", "Parent Scrap qty", "Order Unit", "Prod.Hier",
    "MRP Controller", "Order Type", "Control key", "Tool Number", "GPL Code",
    "Plant", "Actual Operator Id", "Prodn Supervisor", "Entered Good Qty",
    "Entered Scrap Qty", "Entered UoM", "Person responsible", "Name of the person responsible"
]
