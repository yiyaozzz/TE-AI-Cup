import cv2
import numpy as np
import os


def compute_gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=-1)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag


def find_and_draw_bounding_boxes(img, binary_img):
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collecting bounding boxes for all contours
    boxes = [cv2.boundingRect(contour) for contour in contours]

    # Sort boxes by their y-coordinate, then by their x-coordinate
    boxes.sort(key=lambda x: (x[1], x[0]))

    merged_boxes = merge_boxes_based_on_proximity(
        boxes, proximity_threshold=5)

    for box in merged_boxes:
        w, x, y, h = box

        print("x" + str(x))
        print("w" + str(w))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


def merge_boxes_based_on_proximity(boxes, proximity_threshold):
    merged_boxes = []
    current_merge = list(boxes[0]) if boxes else None

    for box in boxes[1:]:
        if current_merge:
            # Check if the current box is close enough to the merge
            if box[0] - (current_merge[0] + current_merge[2]) <= proximity_threshold:
                # Update the current merge to include  box
                new_width = box[0] + box[2] - current_merge[0]
                current_merge[2] = new_width  # Update width
                # Update height if necessary
                current_merge[3] = max(current_merge[3], box[3])
            else:
                # If not close, add the current merge to merged_boxes and start a new merge
                merged_boxes.append(tuple(current_merge))
                current_merge = list(box)
        else:
            current_merge = list(box)

    # Add the last merge if it exists
    if current_merge:
        merged_boxes.append(tuple(current_merge))

    return merged_boxes


def morphological_operations(binary_img):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(4)))
    dilated = cv2.dilate(binary_img, kernel, iterations=1)

    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    # eroded = cv2.erode(dilated, vertical_kernel, iterations=1)

    return dilated


# def find_and_draw_bounding_boxes(img, binary_img):
#     contours, _ = cv2.findContours(
#         binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return img


img = cv2.imread('img11.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

test = compute_gradient(gray_blurred)

_, binary_img = cv2.threshold(gray_blurred, 177, 255, cv2.THRESH_BINARY_INV)

morphed_img = morphological_operations(binary_img)

# contours, _ = cv2.findContours(
#     morphed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# img_with_boxes = img.copy()
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
img_with_boxes_before = img.copy()
find_and_draw_bounding_boxes(img_with_boxes_before, binary_img)
img_with_boxes_after = img.copy()
find_and_draw_bounding_boxes(img_with_boxes_after, morphed_img)


cv2.imshow('Original Image', img)
cv2.imshow('Thresholded Image', binary_img)
cv2.imshow('REmove', morphed_img)
# cv2.imshow('Words Bounded', img_with_boxes)
cv2.imshow('Bounding Boxes Before Merging', img_with_boxes_before)
cv2.imshow('Bounding Boxes After Merging', img_with_boxes_after)

cv2.waitKey(0)
cv2.destroyAllWindows()
