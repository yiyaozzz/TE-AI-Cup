import cv2
from io import BytesIO
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import pandas as pd
import re
import datetime
from main.variables import COLUMNHEADING


def crop_image(image):

    crop_lines = (580,)
    cropped_image = image[:crop_lines[0], :]
    return cropped_image


def run_ocr_on_image(cropped_image, model):

    image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    img_byte_arr = BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    image_cv = DocumentFile.from_images([img_byte_arr])
    result = model(image_cv)
    recognized_text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join([word.value for word in line.words])
                recognized_text += line_text + "\n"
    return recognized_text.strip()


def parse_data_from_ocr_text(ocr_text):

    data = {}
    patterns = {
        "Prod Order": r"Production Order: (\d+)",
        "Op. Good Qty": r"Production Order Qty: (\d+)",
        "UoM": r"(\bPC\b)",
        "Material Number": r"Material: ([\w-]+)",
        "Material Description": r"Description: (.+)",
        "Order Type": r"Order Type: (\w+)",
        "Plant": r"Plant / Business (\d+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, ocr_text)
        if match:
            data[key] = match.group(1)
        else:
            data[key] = ""

    return data


def write_to_excel(data, excel_path, row_count):

    # Ensure the directory exists
    directory = os.path.dirname(excel_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Load existing data or create a new DataFrame if the file doesn't exist
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        # Convert all columns to object type to avoid dtype issues
        df = df.astype('object')
    else:
        df = pd.DataFrame(columns=COLUMNHEADING)
        df = df.astype('object')

    # Adjust df to match the expected number of rows
    if len(df) < row_count:
        # If df has fewer rows than needed, append missing rows
        additional_rows = pd.DataFrame([pd.Series(
            [None]*len(COLUMNHEADING), index=COLUMNHEADING)] * (row_count - len(df)))
        df = pd.concat([df, additional_rows], ignore_index=True)
    elif len(df) > row_count:
        #  If df has more rows than needed, truncate the excess
        df = df.iloc[:row_count]

# Specific data equalization logic
    uom_value = data.get('UoM', "")
    good_qty_value = data.get('Op. Good Qty', "")
    # Apply the data to all required rows for specified columns
    for col in ['UoM', 'Order Unit', 'Entered UoM']:
        df[col] = [uom_value] * row_count

    for col in ['Op. Good Qty', 'Parent Good qty', 'Entered Good Qty']:
        df.loc[0, col] = good_qty_value

    # Overwrite data directly in the first row and replicate it to all required rows
    for key in ['Prod Order', 'Material Number', 'Material Description', 'Order Type', 'Plant']:
        # Overwrite and fill down the column
        df[key] = [data.get(key, "")] * row_count

    # Populate 'Posting date' and 'Entry Date' with today's date
    # Get today's date in the specified format
    today_date = datetime.datetime.now().strftime("%m/%d/%y")
    df['Posting date'] = [today_date] * row_count
    df['Entry Date'] = [today_date] * row_count

    df.to_excel(excel_path, index=False)


def process_image_for_ocr(image_path, excel_path, row_count):

    model = ocr_predictor(pretrained=True)
    image = np.copy(cv2.imread(image_path))
    cropped_image = crop_image(image)
    recognized_text = run_ocr_on_image(cropped_image, model)
    #print("OCR results:")
    #print(recognized_text)
    parsed_data = parse_data_from_ocr_text(recognized_text)
    write_to_excel(parsed_data, excel_path, row_count)
    #print(f"Data successfully written to Excel at {excel_path}")
    # Testing
# Example path to the image
# image_path = '/Users/zyy/Desktop/TE/excel_update/first_page_dataset/7855.png'
# process_image_for_ocr(image_path)
