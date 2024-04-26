import cv2
from io import BytesIO
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import pandas as pd
import re

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
    # Regex patterns to extract data
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
            data[key] = ""  # Default to empty if no match found

    # Handling Material Description specifically to include the next line number if it's numeric
    description_match = re.search(r"Description: (.+?)(?=\n)", ocr_text)
    if description_match:
        description_text = description_match.group(1).strip()
        # Find the number following the description, which is the next line
        next_number_match = re.search(r"(\d+)\nOrder Type:", ocr_text)
        if next_number_match:
            description_text += " " + next_number_match.group(1)
        data["Material Description"] = description_text
    else:
        data["Material Description"] = ""
    return data

def write_to_excel(data, excel_path):
    columns = ["Work Center", "Operation", "Scrap Code", "Scrap Description",
               "Op. Good Qty", "Op. Scrap Qty", "UoM", "PPM__________",
               "Posting date", "Entry Date", "Prod Order", "Material Number",
               "Material Description", "Parent Good qty", "Parent Scrap qty",
               "Order Unit", "Order Type", "Plant", "Entered Good Qty",
               "Entered Scrap Qty", "Entered UoM"]
    df = pd.DataFrame([[
        "", "", "", "", data.get("Op. Good Qty", ""),
        "", data.get("UoM", ""), "", "", "", data.get("Prod Order", ""), data.get("Material Number", ""),
        data.get("Material Description", ""), data.get("Op. Good Qty", ""),  # Parent good qty is same as op good qty
        "", data.get("UoM", ""), data.get("Order Type", ""), data.get("Plant", ""),
        data.get("Op. Good Qty", ""), "", data.get("UoM", "")  # Entered quantities and UoM are the same
    ]], columns=columns)
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df.to_excel(excel_path, index=False)
    print(f"Data successfully written to Excel at {excel_path}")
    
def process_image_for_ocr(image_path):
    model = ocr_predictor(pretrained=True)
    image = np.copy(cv2.imread(image_path))
    cropped_image = crop_image(image)
    recognized_text = run_ocr_on_image(cropped_image, model)
    print("OCR results:")
    print(recognized_text)

    parsed_data = parse_data_from_ocr_text(recognized_text)

    # Creating the ocrOutput directory if it does not exist
    output_dir = os.path.join(os.getcwd(), "ocrOutput")
    os.makedirs(output_dir, exist_ok=True)

    # Define the paths for saving the OCR results text and Excel files
    base_name = os.path.basename(image_path).replace(".png", "")
    text_file_path = os.path.join(output_dir, f"{base_name}.txt")
    excel_file_path = os.path.join(output_dir, f"{base_name}.xlsx")
                                   
    # Saving the OCR results to a text file
    with open(text_file_path, 'w') as f:
        f.write(recognized_text)
    print(f"OCR results saved to {text_file_path}")

    # Write the results to an Excel file
    write_to_excel(parsed_data, excel_file_path)

# Testing
# Example path to the image
# image_path = '/Users/zyy/Documents/GitHub/TE-AI-Cup/500000294405_pages/page_1.png'
# process_image_for_ocr(image_path)