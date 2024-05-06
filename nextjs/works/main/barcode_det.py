import cv2
import numpy as np


def display_image(window_name, image):
    """Displays an image until a key is pressed."""
    cv2.imshow(window_name, image)
    # cv2.waitKey(0)  # Wait for a key press to continue
    # cv2.destroyAllWindows()


def isolate_high_contrast_areas(image):
    """Applies blurring, thresholding, and morphological operations to highlight areas of high contrast."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # display_image("Blurred Image", blurred)

    # Use adaptive thresholding to create a binary image where high contrast areas are white
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # display_image("Threshold Image", thresh)

    # Morphological closing to bridge gaps in barcode lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # display_image("Morphologically Closed Image", closed)

    # Erosion followed by dilation to clean up the image
    eroded = cv2.erode(closed, None, iterations=5)
    # display_image("Eroded Image", eroded)
    dilated = cv2.dilate(eroded, None, iterations=7)
    # display_image("Dilated Image", dilated)

    return dilated


"""
# Direct Implement
def erase_barcodes(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    display_image("Original Image", image)

    processed_image = isolate_high_contrast_areas(image)

    # Convert the processed image back to a binary image where barcodes are black for better contour detection
    ret, binary_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    display_image("Binary Image for Contour Detection", binary_image)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # For displaying colored contours on grayscale image
    margin = 4
    
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        if w > 99 and h > 5:  # Conditions for identifying barcodes based on expected dimensions
            cv2.rectangle(color_image, (x - margin, y - margin), (x + w - margin, y + h + margin), (0, 255, 0), 2)
            cv2.rectangle(image, (x - margin, y - margin), (x + w - margin , y + h + margin), (255, 255, 255), -1)
    display_image("Barcodes Detected", color_image)  # Show where barcodes are detected

    cv2.imwrite('modified_image.png', image)
    display_image('Image without Barcodes', image)  # Display final image with barcodes erased

# Example usage
erase_barcodes('img4.png')
"""


def erase_barcodes_from_image(image):
    """Detects and obscures barcodes by processing a provided image array."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    processed_image = isolate_high_contrast_areas(gray)
    ret, binary_image = cv2.threshold(
        processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin = 4  # Set a margin for erasing around the detected barcode
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        if w > 99 and h > 5:  # Filter contours based on size
            cv2.rectangle(image, (x - margin, y - margin),
                          (x + w+2, y + h + margin), (255, 255, 255), -1)
    return image


'''
# Testing with image loading
if __name__ == "__main__":
    img_path = '/Users/zyy/Documents/GitHub/TE-AI-Cup/500000294405_pages/page_10.png'
    img = cv2.imread(img_path)
    img_without_barcodes = erase_barcodes_from_image(img)
    cv2.imshow('Image without Barcodes', img_without_barcodes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
