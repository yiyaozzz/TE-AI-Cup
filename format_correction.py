import cv2
import numpy as np
from pdf2image import convert_from_path
import os

# Convert PDF to Image


def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

# Rotate the Portrait Format Into Landscape


def rotate_image_to_landscape(image):
    image_np = np.array(image)
    # Convert RGB to BGR format
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]

    # Rotate if in portrait mode
    if h > w:
        image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)

    return image_np


def save_images(images, output_dir, base_filename='image'):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each image to the output directory
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{base_filename}_{i+1}.jpg")
        cv2.imwrite(output_path, image)


def process_pdf_to_landscape(pdf_path, output_dir):
    # Processes a PDF file, ensuring all pages are in landscape orientation
    images = convert_pdf_to_images(pdf_path)
    landscape_images = [rotate_image_to_landscape(image) for image in images]
    save_images(landscape_images, output_dir)

    return landscape_images


process_pdf_to_landscape(
    "/Users/pravin/Desktop/TE_Comp/TE-AI-Cup/data/500000066049.pdf", "/Users/pravin/Desktop/TE_Comp/TE-AI-Cup/result")
