from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from format_correction import process_and_crop_pdf
import os
import cv2


def run_ocr_on_images(image_paths, model):
    recognized_text = ""
    for image_path in image_paths:
        result = model(DocumentFile.from_images(image_path))
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ' '.join([word.value for word in line.words])
                    recognized_text += line_text + "\n"
    return recognized_text


def main(pdf_path, header_dir, table_dir, output_text_path, first_image_crop_lines, rest_image_crop_line):
    os.environ["USE_TORCH"] = "1"

    process_and_crop_pdf(pdf_path, header_dir, table_dir,
                         first_image_crop_lines, rest_image_crop_line)

    model = ocr_predictor(pretrained=True)

    header_image_path = os.path.join(header_dir, "header.jpg")
    # table_images_paths = [os.path.join(table_dir, filename) for filename in os.listdir(table_dir) if filename.startswith("table")]

    # Perform OCR on header
    header_text = run_ocr_on_images([header_image_path], model)

    # Perform OCR on table parts
    # table_text = run_ocr_on_images(table_images_paths, model)

    # Combine header and table text
    # recognized_text = header_text + "\n" + table_text

    # Save the recognized text to a file
    with open(output_text_path, 'w') as text_file:
        text_file.write(header_text)

    print(f"OCR results saved to {output_text_path}")


if __name__ == "__main__":
    pdf_path = "/data/Production Sheet/500000261553.pdf"
    # Specify directories for header and table parts of cropped images
    header_dir = "template/header"
    table_dir = "template/table"
    output_text_path = "result/result.txt"
    # Crop lines for the first image
    first_image_crop_lines = (540, 1330)
    # Crop line for the rest of the images
    rest_image_crop_line = (0, 0)
    main(pdf_path, header_dir, table_dir, output_text_path,
         first_image_crop_lines, rest_image_crop_line)
