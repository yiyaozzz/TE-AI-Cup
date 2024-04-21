import cv2
import numpy as np
import os


def apply_morphology(img):
    # Create horizontal kernel and apply morphological operations
    kernel_width = np.ones((1, 6), np.uint8)  # More width emphasis
    kernel_height = np.ones((6, 1), np.uint8)  # More height emphasis

    # Morphological closing (dilate followed by erode)
    closing_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_width)

    # Morphological opening (erode followed by dilate)
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel_height)

    return opening_img


def make_border_white(img, border_size=1):
    # Set top and bottom border to white
    img[:border_size, :] = 255
    img[-border_size:, :] = 255

    # Set left and right border to white
    img[:, :border_size] = 255
    img[:, -border_size:] = 255

    return img


# Load the image and convert it to grayscale
img = cv2.imread('Cells/cropped_row2_cell5.png', cv2.IMREAD_GRAYSCALE)
img = make_border_white(img, border_size=1)

# Apply thresholding
_, thresh_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
thresh_img = cv2.GaussianBlur(thresh_img, (5, 5), 0)
_, thresh_img = cv2.threshold(thresh_img, 70, 255, cv2.THRESH_BINARY)

# Apply morphology
morph_img = apply_morphology(thresh_img)
cv2.imshow("Morphological Processing", morph_img)

# Define contours and bounding boxes for further processing
contours, hierarchy = cv2.findContours(
    morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c)
                  for c in contours if cv2.contourArea(c) > 500]
# Sort by y coordinate
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

# Visualize bounding boxes
img_with_bounds = img.copy()
for x, y, w, h in bounding_boxes:
    cv2.rectangle(img_with_bounds, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Words with Bounding Boxes", img_with_bounds)

cv2.waitKey(0)
cv2.destroyAllWindows()
