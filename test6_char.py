import cv2
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread('img13.png')
# cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Binary', binary_img)
cv2.waitKey(0)
# Find Canny edges
edged = cv2.Canny(binary_img, 30, 200)
cv2.imshow('Edge', edged)
cv2.waitKey(0)

kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow('Dilated Edge', dilated)
cv2.waitKey(0)
# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(dilated,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


img_with_boxes = image.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)


def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
    ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
    return thresh


cv2.imshow('Boxed', img_with_boxes)
# cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))
