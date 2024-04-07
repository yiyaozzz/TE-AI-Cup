from ocr_det import process_pdf
from validation import validate_images
import process_results


def run_pipeline(pdf_path):

    images_dir = process_pdf(pdf_path)

    labels_list = validate_images(images_dir)

    process_results.process(label_list=labels_list)
