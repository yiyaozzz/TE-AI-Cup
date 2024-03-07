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


def crop_image(image, crop_lines, part):
    if part == 'top':
        cropped_image = image[:crop_lines[0], :]
    elif part == 'middle':
        cropped_image = image[crop_lines[0]:crop_lines[1], :]
    elif part == 'bottom':
        cropped_image = image[crop_lines[1]:, :]
    else:
        raise ValueError("Invalid part_to_save value. Choose 'top', 'middle', or 'bottom'.")
    return cropped_image

# Save cropped parts of first image and the middle parts of the rest of the images
def save_cropped_images(images, output_dirs, first_image_crop_lines, rest_image_crop_line):
    header_dir, table_dir = output_dirs

    # Ensure output directories exist
    if not os.path.exists(header_dir):
        os.makedirs(header_dir)
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    # Process the first image
    first_image = images[0]
    header_part = crop_image(first_image, first_image_crop_lines, 'top')
    table_part_first = crop_image(first_image, first_image_crop_lines, 'middle')
    cv2.imwrite(os.path.join(header_dir, "header.jpg"), header_part)
    cv2.imwrite(os.path.join(table_dir, "table_first.jpg"), table_part_first)
    
    # Process the rest of the images
    for i, image in enumerate(images[1:], start=1):
        table_part_rest = crop_image(image, rest_image_crop_line, 'middle')
        cv2.imwrite(os.path.join(table_dir, f"table_rest_{i}.jpg"), table_part_rest)

# Convert image into landscape and save the cropped images
def process_images(pdf_path, output_dirs, first_image_crop_lines, rest_image_crop_line):
    images = convert_pdf_to_images(pdf_path)
    landscape_images = [rotate_image_to_landscape(image) for image in images]
    save_cropped_images(landscape_images, output_dirs, first_image_crop_lines, rest_image_crop_line)

# Specify directories for header and table parts
output_header_dir = "template/header"
output_table_dir = "template/table"
output_dirs = (output_header_dir, output_table_dir)

# Crop lines for the first image and the rest of the images
first_image_crop_lines = (540, 1330)  
rest_image_crop_line = (220, 1390)     

# Process the PDF, cropping as required
process_images(
    pdf_path="data/Production Sheet/500000261553.pdf",
    output_dirs=output_dirs,
    first_image_crop_lines=first_image_crop_lines,
    rest_image_crop_line=rest_image_crop_line
)
