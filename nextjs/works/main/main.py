from main.pdf_converter import convert_pdf_to_png
import os
from main.ocr_det import take_input
from main.tt import process_images_in_folder
from main.firstWord import detect_first_word
from main.yolo_pred import track_object
import time
from main.popSheet import process_files
from main.validate import finalVal


def process_pdf_or_folder(input_data, is_file=False):
    fileID = os.path.basename(input_data)
    print("Processing input data:", input_data)

    if is_file:  # Handling a single uploaded file
        file_path = input_data
        images = convert_pdf_to_png(file_path)
        # ocr_detection(file_path/page_1)
        for image_path in images:
            print("Image path:", image_path)
            take_input(image_path, fileID)
        # os.remove(file_path)  # Optional: Remove the PDF file after processing
    else:  # Handling a folder path
        if not os.path.isdir(input_data):
            return "Invalid directory path."
        for filename in os.listdir(input_data):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_data, filename)
                images = convert_pdf_to_png(pdf_path)
                for image_path in images:
                    take_input(image_path, fileID)

    # After all images are processed, handle based on columns
    tempTables_path = f"tempTables_{fileID}"

    if not os.path.exists(tempTables_path):
        print("No tempTables directory found.")
        return

    # Iterate over each folder in tempTables
    for folder_name in os.listdir(tempTables_path):
        folder_path = os.path.join(tempTables_path, folder_name)
        for row_name in os.listdir(folder_path):  # Iterate over each row
            row_path = os.path.join(folder_path, row_name)
            for column_name in os.listdir(row_path):
                column_path = os.path.join(row_path, column_name)
                images = [os.path.join(column_path, f) for f in os.listdir(
                    column_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                if column_name in ["column_1", "column_2"]:
                    for image in images:
                        detect_first_word(
                            image, fileID)
                elif column_name in ["column_3", "column_4"]:
                    print(
                        f"Processing folder for object tracking: {column_name}")
                    print(column_path)

                    process_images_in_folder(column_path)
                    track_object(column_path, f'finalOutput_{fileID}')

    process_files(f'finalOutput_{fileID}', fileID)

    return "All PDFs have been processed."