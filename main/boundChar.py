import cv2
import numpy as np


def process_image(path):
    # Read Input image
    inputImage = cv2.imread(path)

    # Deep copy for results:
    inputImageCopy = inputImage.copy()

    # Convert BGR to grayscale:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Threshold via Otsu:
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv2.imshow("Otsu Threshold", binaryImage)
    #cv2.waitKey(0)




    # Flood-fill border, seed at (0,0) and use black (0) color:
    cv2.floodFill(binaryImage, None, (0, 0), 0)
    #cv2.imshow("Flood Fill", binaryImage)
    #cv2.waitKey(0)


    # Get each bounding box
    # Find the big contours/blobs on the filtered image:
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the outer bounding boxes (no children):
    for _, c in enumerate(contours):

        # Get the bounding rectangle of the current contour:
        boundRect = cv2.boundingRect(c)

        # Get the bounding rectangle data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Estimate the bounding rect area:
        rectArea = rectWidth * rectHeight

        # Set a min area threshold
        minArea = 8

        # Filter blobs by area:
        if rectArea > minArea:

            # Draw bounding box:
            color = (0, 255, 0)
            cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                        (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
            cv2.imshow("Bounding Boxes", inputImageCopy)

            # Crop bounding box:
            currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
            cv2.imshow("Current Crop", currentCrop)
            cv2.waitKey(0)

# Test
process_image("/Users/zyy/Desktop/TE/Bounding/1_Words.png")