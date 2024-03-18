import cv2
import pytesseract
from pytesseract import Output
import pandas as pd


def highlight_columns(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    d = pytesseract.image_to_data(gray, output_type=Output.DICT)

    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top']
                        [i], d['width'][i], d['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Highlighted Columns', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'template/table/table_first.jpg'
highlight_columns(image_path)
