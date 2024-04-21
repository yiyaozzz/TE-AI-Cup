import cv2
import numpy as np
import os
from barcode_det import erase_barcodes_from_image


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_and_transform_largest_table(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    max_area = 0
    max_width = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)

        if aspect_ratio > 1.5 and area > 750:
            if area > max_area:
                max_area = area
                best_cnt = cnt
                max_width = w

    if best_cnt is not None:
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_ordered = order_points(box)

        (tl, tr, br, bl) = rect_ordered
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_width = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_height = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect_ordered, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    return image

# def extract_and_transform_largest_table(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(
#         gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     contours, _ = cv2.findContours(
#         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     max_area = 0
#     best_cnt = None
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#         if area > max_area and len(approx) == 4:
#             max_area = area
#             best_cnt = approx
#     if best_cnt is not None:
#         rect = order_points(best_cnt.reshape(4, 2))
#         (tl, tr, br, bl) = rect
#         widthA = np.linalg.norm(br - bl)
#         widthB = np.linalg.norm(tr - tl)
#         max_width = max(int(widthA), int(widthB))
#         heightA = np.linalg.norm(tr - br)
#         heightB = np.linalg.norm(tl - bl)
#         max_height = max(int(heightA), int(heightB))
#         dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1,
#                        max_height - 1], [0, max_height - 1]], dtype="float32")
#         M = cv2.getPerspectiveTransform(rect, dst)
#         warped = cv2.warpPerspective(image, M, (max_width, max_height))
#         return warped
#     return image


def sort_contours(contours, method="top-to-bottom"):
    reverse = False
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    i = 0 if method == "left-to-right" or method == "right-to-left" else 1
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(
        *sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    return (contours, bounding_boxes)


# Load and preprocess the image
file = 'img15.png'
img = cv2.imread(file)
img = erase_barcodes_from_image(img)  # First erase barcodes
# Then extract and transform the largest table
img = extract_and_transform_largest_table(img)
cv2.imshow("image", img)
cv2.waitKey(0)

# Convert to grayscale for further processing
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh, img_bin = cv2.threshold(
    img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = 255 - img_bin  # Inverting the image

# Further image processing
kernel_len = np.array(img).shape[1]//100
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Apply morphological operations
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

cv2.imshow("vertical", vertical_lines)
cv2.imshow("horizontal", horizontal_lines)
cv2.waitKey(0)

img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(
    img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find and sort contours for cell extraction
contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = []
# Filter out noise cells
min_area = 300
min_width = 20
min_height = 20
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > min_area and w > min_width and h > min_height:
        filtered_contours.append(cnt)

contours, bounding_boxes = sort_contours(filtered_contours, "top-to-bottom")

# Group contours into rows and columns
rows = []
current_row = []
prev_y = -1
# Store center of the previous contour
prev_center = None

for contour, box in zip(contours, bounding_boxes):
    x, y, w, h = box
    # Calculate center of the current contour
    center = (x + w // 2, y + h // 2)
    # Adjust this threshold for distance between center and next row
    if prev_y == -1 or abs(center[1] - prev_center[1]) < h * 0.2:
        current_row.append(contour)
    else:
        rows.append(current_row)
        current_row = [contour]
    prev_y = y
    prev_center = center

# Add the last row
if current_row:
    rows.append(current_row)

# Crop and save each cell
base_folder = "./cropped_cells"
os.makedirs(base_folder, exist_ok=True)

for row_idx, row in enumerate(rows, start=1):
    row_folder = os.path.join(base_folder, f"row_{row_idx}")
    os.makedirs(row_folder, exist_ok=True)
    column_contours, _ = sort_contours(row, "left-to-right")
    # Specify columns to save
    columns_to_save = [1, 2, 4, 5]
    for cell_idx, cnt in enumerate(column_contours):
        if cell_idx + 1 in columns_to_save:
            x, y, w, h = cv2.boundingRect(cnt)
            column_folder = os.path.join(row_folder, f"column_{cell_idx+1}")
            os.makedirs(column_folder, exist_ok=True)
            cropped_cell = img[y:y+h, x:x+w]
            filename = os.path.join(column_folder, f"cell_{cell_idx+1}.png")
            cv2.imwrite(filename, cropped_cell)
            print(f"Saved: {filename}")
