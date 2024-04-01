import cv2
import numpy as np

# Set image path
path = ""
fileName = "datasets/test/ew/test.png"

# Read Input image
inputImage = cv2.imread(path+fileName)

# Deep copy for results:
inputImageCopy = inputImage.copy()

# Convert BGR to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Threshold via Otsu:
threshValue, binaryImage = cv2.threshold(
    grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.floodFill(binaryImage, None, (0, 0), 0)
cv2.imshow("Bounding Boxes", binaryImage)

contours, hierarchy = cv2.findContours(
    binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for _, c in enumerate(contours):

    # Get the bounding rectangle of the current contour:
    boundRect = cv2.boundingRect(c)

    # Get the bounding rectangle data:
    rectX, rectY, rectWidth, rectHeight = boundRect

    # Estimate the bounding rect area:
    rectArea = rectWidth * rectHeight

    # Set minimum area and size thresholds
    minArea = 8
    minWidth = 8
    minHeight = 8

    # Filter blobs by area and size:
    if rectArea > minArea and rectWidth >= minWidth and rectHeight >= minHeight:

        # Draw bounding box:
        color = (0, 255, 0)
        cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                      (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
        cv2.imshow("Bounding Boxes", inputImageCopy)
        # Crop bounding box:
        currentCrop = inputImage[rectY:rectY+rectHeight, rectX:rectX+rectWidth]
        # cv2.imshow("Current Crop", currentCrop)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
