import cv2
import numpy as np
from pdf2image import convert_from_path

#Convert PDF to Image
def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

#Rotate the Portrait Format Into Landscape
def rotate_image_to_landscape(image):
    image_np = np.array(image)
    # Convert RGB to BGR format 
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]

    # Rotate if in portrait mode
    if h > w:
        image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
    
    return image_np

def process_pdf_to_landscape(pdf_path):
    #Processes a PDF file, ensuring all pages are in landscape orientation
    images = convert_pdf_to_images(pdf_path)
    landscape_images = [rotate_image_to_landscape(image) for image in images]
    
    return landscape_images
