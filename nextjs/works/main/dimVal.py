import cv2
import numpy as np
from main.api import aapiResult
import time
import os
from main.gapi import apiResult


def process_image(path):
    inputImage = cv2.imread(path)
    if inputImage is None:
        raise ValueError("Image could not be loaded. Please check the path.")

    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    _, binaryImage = cv2.threshold(
        grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.floodFill(binaryImage, None, (0, 0), 0)
    contours, _ = cv2.findContours(
        binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aggregated_text = ""
    for _, c in enumerate(contours):
        boundRect = cv2.boundingRect(c)
        rectX, rectY, rectWidth, rectHeight = boundRect
        rectArea = rectWidth * rectHeight

        minArea = 8
        if rectArea > minArea:
            currentCrop = inputImage[rectY:rectY +
                                     rectHeight, rectX:rectX + rectWidth]
            cropped_image_path = save_cropped_image(currentCrop)
            text_from_api = aapiResult(cropped_image_path)
            # Ensure text_from_api is not None and is a string before concatenating
            if text_from_api:

                aggregated_text += text_from_api
                aggregated_text = aggregated_text.strip()

    return aggregated_text


def process_image_gapi(path):
    inputImage = cv2.imread(path)
    if inputImage is None:
        raise ValueError("Image could not be loaded. Please check the path.")

    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    _, binaryImage = cv2.threshold(
        grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.floodFill(binaryImage, None, (0, 0), 0)
    contours, _ = cv2.findContours(
        binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aggregated_text = ""
    for _, c in enumerate(contours):
        boundRect = cv2.boundingRect(c)
        rectX, rectY, rectWidth, rectHeight = boundRect
        rectArea = rectWidth * rectHeight

        minArea = 8
        if rectArea > minArea:
            currentCrop = inputImage[rectY:rectY +
                                     rectHeight, rectX:rectX + rectWidth]
            cropped_image_path = save_cropped_image(currentCrop)
            text_from_api = apiResult(cropped_image_path)
            # Ensure text_from_api is not None and is a string before concatenating
            if text_from_api is None:
                aggregated_text = None
                # Safely concatenate valid strings
            else:
                aggregated_text += text_from_api
                aggregated_text = aggregated_text.strip()

    return aggregated_text


def save_cropped_image(crop_img):
    # Generate a unique filename for each crop
    filename = f"cropped_{int(time.time() * 1000)}.jpg"
    save_path = f"temp_cropped_images/{filename}"
    if not os.path.exists('temp_cropped_images'):
        os.makedirs('temp_cropped_images')
    cv2.imwrite(save_path, crop_img)
    return save_path


def delete_cropped_images(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All cropped images have been deleted.")
    else:
        print("Directory does not exist.")


def dimValPred(file_path):
    print(file_path)
    result = process_image(file_path)

    if result is None:
        result = process_image_gapi(file_path)
    if result is not None:
        print("dimres " + result)
    return result


# delete_cropped_images('temp_cropped_images')
