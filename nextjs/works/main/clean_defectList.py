import pandas as pd

def clean_sort_and_highlight_excel(file_path, output_file):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Remove rows where both 'Scrap Code' and 'Scrap Description' are empty
    df = df.dropna(subset=['Scrap Code', 'Scrap Description'], how='all')

    # Remove duplicates based on 'Scrap Code' and 'Scrap Description'
    df = df.drop_duplicates(subset=['Scrap Code', 'Scrap Description'])

    # Sort the DataFrame alphabetically by 'Scrap Description'
    df = df.sort_values(by='Scrap Description')

    # Find duplicates in 'Scrap Code'
    duplicates = df.duplicated(subset='Scrap Code', keep=False)

    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')

    # Access the XlsxWriter workbook and worksheet objects
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']

    # Define a format to use for highlighting cells
    highlight_format = workbook.add_format({'bg_color': 'yellow'})

    # Apply conditional formatting to the duplicate rows
    for row in range(1, len(duplicates) + 1):  # Adjust range for Excel indexing
        if duplicates.iloc[row - 1]:  # Check if row is duplicate
            worksheet.set_row(row, None, highlight_format)

    # Close the Pandas Excel writer and output the Excel file
    writer.close()

    # Confirmation message
    print(f"Cleaned data has been saved and highlighted to {output_file}")

# Specify the path to your Excel file and the output file name
input_file_path = '/Users/zyy/Desktop/TE/latest/defect/cleaned_data.xlsx'
output_file_path = '/Users/zyy/Desktop/TE/latest/defect/final_defect_list.xlsx'

# Run the function with the specified file paths
clean_sort_and_highlight_excel(input_file_path, output_file_path)
