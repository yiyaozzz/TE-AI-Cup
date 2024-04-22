import os
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from tt import resize_and_pad


def correct_orientation(image_path):
    with Image.open(image_path) as img:
        img = resize_and_pad(image_path, target_width=640, target_height=360)

        ocr_data = pytesseract.image_to_osd(
            img, output_type=pytesseract.Output.DICT)
        rotation_angle = ocr_data['rotate']
        print(f"Detected rotation for {image_path}: {rotation_angle} degrees")

        if rotation_angle != 0:
            corrected_img = img.rotate(-rotation_angle, expand=True)
            corrected_img.save(image_path)
            print(f"Corrected and saved {image_path}")
# def correct_orientation(image_path):
#     with Image.open(image_path) as img:
#         img = preprocess_image_for_ocr(img)
#         try:
#             ocr_data = pytesseract.image_to_osd(
#                 img, output_type=pytesseract.Output.DICT)
#             rotation_angle = ocr_data['rotate']
#             print(
#                 f"Detected rotation for {image_path}: {rotation_angle} degrees")

#             if rotation_angle != 0:
#                 corrected_img = img.rotate(-rotation_angle, expand=True)
#                 corrected_img.save(image_path)
#                 print(f"Corrected and saved {image_path}")
#         except pytesseract.TesseractError as e:
#             print(f"Error processing {image_path}: {e}")


def process_images(base_path):
    for page_folder in os.listdir(base_path):
        if page_folder.startswith("page_"):
            page_path = os.path.join(base_path, page_folder)
            for row_folder in os.listdir(page_path):
                if row_folder.startswith("row_"):
                    row_path = os.path.join(page_path, row_folder)
                    for col_folder in os.listdir(row_path):
                        if col_folder == "column_3":
                            col_path = os.path.join(row_path, col_folder)
                            image_files = [f for f in os.listdir(col_path) if f.lower().endswith(
                                ('.png', '.jpg', '.jpeg', '.bmp'))]
                            if image_files:
                                first_image = image_files[0]
                                first_image_path = os.path.join(
                                    col_path, first_image)
                                correct_orientation(first_image_path)


base_path = 'finalOutput'
process_images(base_path)
