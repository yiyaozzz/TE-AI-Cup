import cv2
import pytesseract
import os
import re
word_counter = 1


def detect_first_word(image_path):
    path_parts = image_path.split(os.sep)
    page_pattern = re.compile(r'^page_\d+$')
    for part in path_parts:
        if page_pattern.match(part):
            pth = part
    global word_counter
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT, config=custom_config)

    output_dir = "finalOutput"
    os.makedirs(output_dir, exist_ok=True)

    for i, word in enumerate(details['text']):
        if word.strip():
            x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
            cropped_img = img[y:y+h+3, x:x+w+3]
            word_counter += 1
            output_dir = os.path.join(output_dir, pth)
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(
                output_dir, f"cell_{word_counter}.png")
            cv2.imwrite(output_filename, cropped_img)
            print(f"Saved cropped word '{word}' to '{output_filename}'")
            if word == 2:
                word = 1
            break
