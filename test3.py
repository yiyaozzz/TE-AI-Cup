import cv2
import numpy as np

# Set image path
path = ""
fileName = "img5.png"

# Read Input image
inputImage = cv2.imread(path+fileName)

# Deep copy for results:
inputImageCopy = inputImage.copy()

# Convert BGR to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to smooth out the image
blurredImage = cv2.GaussianBlur(grayscaleImage, (5, 5), 0)

# Threshold via Otsu after blurring:
_, binaryImage = cv2.threshold(
    blurredImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to separate text more effectively
kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
opening = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and hierarchy
contours, hierarchy = cv2.findContours(
    opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i, c in enumerate(contours):
    # Only consider child contours (assuming text is within tables)
    if hierarchy[0][i][3] != -1:  # Check if there's a parent contour
        boundRect = cv2.boundingRect(c)
        rectX, rectY, rectWidth, rectHeight = boundRect

        # Estimate the bounding rect area:
        rectArea = rectWidth * rectHeight

        # Set minimum area and size thresholds for text
        minArea = 20  # Smaller area for text
        minWidth = 15
        minHeight = 15

        if rectArea > minArea and rectWidth >= minWidth and rectHeight >= minHeight:
            # Draw bounding box around text:
            color = (0, 255, 0)
            cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                          (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

# Show results
cv2.imshow("Detected Text", inputImageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
