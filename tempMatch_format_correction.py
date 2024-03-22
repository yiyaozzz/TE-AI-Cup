import cv2
import numpy as np
from pdf2image import convert_from_path
import os

def find_logo(image, logo_template_path, threshold=0.8):
    logo_template = cv2.imread(logo_template_path, 0)  # Read as grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    found = None

    # Loop over the scales of the template
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # Resize the template according to the current scale
        resized = cv2.resize(logo_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # If the resized template is larger than the image, skip this scale
        if resized.shape[0] > image_gray.shape[0] or resized.shape[1] > image_gray.shape[1]:
            continue
        
        # Perform template matching
        res = cv2.matchTemplate(image_gray, resized, cv2.TM_CCOEFF_NORMED)
        # Get the maximum value of the match results
        max_val = np.max(res)
        
        # Check if we have found a new maximum correlation value
        if max_val >= threshold:
            found = True
            break

    # Return whether the logo was found
    return found

def rotate_image_until_logo_matches(image, logo_template_path, threshold=0.8):
    # Check the initial orientation of the image
    if find_logo(image, logo_template_path, threshold):
        return image  # Logo found in the current orientation
    # Rotate the image in 90-degree increments and check for logo
    for _ in range(3):  # We have already checked the initial orientation
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if find_logo(image, logo_template_path, threshold):
            return image  # Logo found in the new orientation
    # If the logo is not found in any orientation, return the original image
    # You can change this behavior to suit your needs
    return image

def rotate_image_to_landscape(image, logo_template_path):
    image_np = np.array(image)
    # Convert RGB to BGR format for OpenCV operations
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Rotate the image until the logo matches the template
    correctly_oriented_image = rotate_image_until_logo_matches(image_np, logo_template_path)
    return correctly_oriented_image

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=200)  # Added dpi for better image quality

def save_images(images, output_dir, base_filename='image'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{base_filename}_{i+1}.jpg")
        cv2.imwrite(output_path, image)

def process_pdf_to_landscape(pdf_path, output_dir, logo_template_path):
    images = convert_pdf_to_images(pdf_path)
    landscape_images = [rotate_image_to_landscape(image, logo_template_path) for image in images]
    save_images(landscape_images, output_dir)

