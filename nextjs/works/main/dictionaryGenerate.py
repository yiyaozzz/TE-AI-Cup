import pandas as pd

# Load the Excel file
df = pd.read_excel('/Users/zyy/Desktop/TE/latest/defect/cleaned_data.xlsx')

# Remove any potential whitespace from column names
df.columns = df.columns.str.strip()

# Drop rows where either 'Scrap Code' or 'Scrap Description' might be NaN
df.dropna(subset=['Scrap Code', 'Scrap Description'], inplace=True)

# Creating a dictionary from the DataFrame
scrap_code_mapping = df.set_index('Scrap Description')['Scrap Code'].to_dict()

# Print each key-value pair on a new line
for description, code in scrap_code_mapping.items():
    print(f"'{description}': '{code}',")
