from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os

def recognize_text_in_image(image_path):
    os.environ["USE_TORCH"] = "1"  

    document = DocumentFile.from_images(image_path)
    model = ocr_predictor(pretrained=True)
    result = model(document)
    recognized_text = ""

    # Extract text
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join([word.value for word in line.words])
                recognized_text += line_text + "\n"

    return recognized_text

if __name__ == "__main__":
    image_path = "/Users/zyy/Desktop/TE/naTest6.png"
    text = recognize_text_in_image(image_path)
    print("Recognized text:\n", text)
