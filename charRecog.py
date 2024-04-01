import cv2
import numpy as np


file_path = 'datasets/test/EW/test.png'
input_image = cv2.imread(file_path)


grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Threshold via Otsu
_, binary_image = cv2.threshold(
    grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

cv2.floodFill(binary_image, None, (0, 0), 0)

contours, hierarchy = cv2.findContours(
    binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

min_area = 5
min_width = 5
min_height = 5

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w * h > min_area and w > min_width and h > min_height:
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

result_path = 'charecter.png'
cv2.imwrite(result_path, input_image)

result_path
# import cv2
# import numpy as np

# file_path = 'cropped_row1_cell2.png'
# input_image = cv2.imread(file_path)

# grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# # Threshold via Otsu
# _, binary_image = cv2.threshold(
#     grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
# )

# cv2.floodFill(binary_image, None, (0, 0), 0)

# contours, hierarchy = cv2.findContours(
#     binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# min_area = 5
# min_width = 5
# min_height = 5
# character_index = 0

# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if w * h > min_area and w > min_width and h > min_height:
#         cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cropped_character = input_image[y:y+h, x:x+w]
#         character_filename = f'char_{character_index}.png'
#         cv2.imwrite(character_filename, cropped_character)
#         character_index += 1

# result_path = 'characters_highlighted.png'
# cv2.imwrite(result_path, input_image)

# print(f"Processed image saved as {result_path}.")
# print(f"{character_index} characters were saved.")
