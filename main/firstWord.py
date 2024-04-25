import cv2
import pytesseract
import os
from tt import process_images_in_folder
from yolo_pred import track_object
word_counter = 1


def create_directory_structure(base_dir, image_path):
    os.makedirs(base_dir, exist_ok=True)

    parts = image_path.split('/')
    page_index = next(i for i, part in enumerate(parts) if 'page_' in part)
    path_to_create = os.path.join(base_dir, *parts[page_index:-1])

    # Create the directory if it does not exist
    os.makedirs(path_to_create, exist_ok=True)

    return path_to_create


def detect_first_word(image_path, base_output_dir="firstWordGen"):
    # Use create_directory_structure to determine the full output path
    output_dir = create_directory_structure(base_output_dir, image_path)

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT, config=custom_config)

    for i, word in enumerate(details['text']):
        if word.strip():
            x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
            cropped_img = img[y:y+h, x:x+w]
            global word_counter
            word_counter += 1
            output_filename = os.path.join(
                output_dir, f"cell_{word_counter}.png")
            cv2.imwrite(output_filename, cropped_img)
            process_images_in_folder(output_dir)
            track_object(output_dir)
            print(f"Saved cropped to '{output_filename}'")
            break  # Breaking after the first word detected and saved


# image_path = 'tempTables/page_1/row_2/column_2/cell_2.png'
# detect_first_word(image_path)
