from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
from format_correction import process_pdf_to_landscape
from PIL import Image
import numpy as np
import cv2
import tempfile
import shutil


def save_landscape_images(images, output_folder):
    # Saves images to disk, useful for debugging or intermediate storage
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_folder, f"page_{i}.jpg"), img)


def main(pdf_path, output_text_path):
    # Ensure the right backend is used
    os.environ["USE_TORCH"] = "1"

    # Create a temporary directory to store landscape images
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Convert PDF to landscape images
        landscape_images = process_pdf_to_landscape(pdf_path)

        # Save the landscape images temporarily and collect their paths
        image_paths = []
        for i, img in enumerate(landscape_images):
            temp_image_path = os.path.join(tmpdirname, f"image_{i}.png")
            cv2.imwrite(temp_image_path, img)
            image_paths.append(temp_image_path)

        # Instantiate the OCR model
        model = ocr_predictor(pretrained=True)

        recognized_text = ""

        # Perform OCR using the paths of the temporarily saved images
        for image_path in image_paths:
            result = model(DocumentFile.from_images(image_path))

            # Extract text
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = ' '.join(
                            [word.value for word in line.words])
                        recognized_text += line_text + "\n"

        # Save the recognized text to a file
        with open(output_text_path, 'w') as text_file:
            text_file.write(recognized_text)

        print(f"OCR results saved to {output_text_path}")


if __name__ == "__main__":
    pdf_path = '/Users/zyy/Documents/GitHub/TE-AI-Cup/data/test01_500000164476.pdf'
    output_text_path = '/Users/zyy/Documents/GitHub/TE-AI-Cup/result/result.txt'
    main(pdf_path, output_text_path)
