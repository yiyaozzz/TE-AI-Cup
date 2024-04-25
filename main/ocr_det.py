import cv2
from io import BytesIO
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os

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

def process_image_for_ocr(image_path):
    model = ocr_predictor(pretrained=True)
    original_image = cv2.imread(image_path)  
    image = np.copy(original_image)
    cropped_image = crop_image(image)  

    recognized_text = run_ocr_on_image(cropped_image, model)
    print("OCR results:")
    print(recognized_text)

    # Creating the ocrOutput directory if it does not exist
    output_dir = os.path.join(os.getcwd(), "ocrOutput")
    os.makedirs(output_dir, exist_ok=True)

    # Define the path for saving the OCR results text file
    text_file_name = os.path.basename(image_path).replace(".png", ".txt")
    text_file_path = os.path.join(output_dir, text_file_name)

    # Saving the results to a text file in the ocrOutput directory
    with open(text_file_path, 'w') as f:
        f.write(recognized_text)
    print(f"OCR results saved to {text_file_path}")

# Testing
# Example path to the image
#image_path = '/Users/zyy/Documents/GitHub/TE-AI-Cup/500000294405_pages/page_1.png'
#process_image_for_ocr(image_path)



'''
#Hard-code the data to excel sheet
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
        data.get("Material Description", ""), data.get("Parent Good qty", ""),
        "", data.get("UoM", ""), data.get("Order Type", ""), data.get("Plant", ""),
        data.get("Entered Good Qty", ""), "", data.get("Entered UoM", "")
    ]], columns=columns)
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df.to_excel(excel_path, index=False)
'''