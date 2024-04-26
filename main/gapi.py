import os
from google.cloud import vision
from PIL import Image

# Set Google API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pravin/Desktop/TE_Comp/TE-AI-Cup/main/service_account_token.json"


def prepare_image(image_path, min_width=1024, min_height=768):

    # Checking dimensions and adding padding if necessary
    with Image.open(image_path) as img:
        width, height = img.size
        if width < min_width or height < min_height:
            new_width = max(min_width, width)
            new_height = max(min_height, height)
            new_img = Image.new("RGB", (new_width, new_height), "white")
            new_img.paste(img, ((new_width - width) //
                          2, (new_height - height) // 2))
            new_img_path = "/tmp/processed_image.png"
            new_img.save(new_img_path, "PNG")
            return new_img_path
        return image_path


def apiResult(path):
    """Detects document features in an image."""
    # Prepare the image
    prepared_image_path = prepare_image(path)

    # Initialize the Google Vision API client
    client = vision.ImageAnnotatorClient()

    # Read the prepared image
    with open(prepared_image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    full_document_text = None

    if not response.full_text_annotation.pages:
        print("No text found: " + path)
        return None
    else:
        # If pages are found, concatenate all the text
        full_text = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    paragraph_text = "".join(
                        [symbol.text for word in paragraph.words for symbol in word.symbols])
                    full_text.append(paragraph_text)

        full_document_text = '\n'.join(full_text)

    return full_document_text if full_document_text else None
# Example usage
# apiResult('finalOutput/page_6/row_4/column_4/1_Number.png')
