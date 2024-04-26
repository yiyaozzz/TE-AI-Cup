import os
from google.cloud import vision
from PIL import Image

# Set Google API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/zyy/Documents/GitHub/TE-AI-Cup/main/service_account_token.json"

def prepare_image(image_path, min_width=1024, min_height=768):
    
    #Checking dimensions and adding padding if necessary
    with Image.open(image_path) as img:
        width, height = img.size
        if width < min_width or height < min_height:
            new_width = max(min_width, width)
            new_height = max(min_height, height)
            new_img = Image.new("RGB", (new_width, new_height), "white")
            new_img.paste(img, ((new_width - width) // 2, (new_height - height) // 2))
            new_img_path = "/tmp/processed_image.png"
            new_img.save(new_img_path, "PNG")
            return new_img_path
        return image_path

def detect_document(path):
    """Detects document features in an image."""
    # Prepare the image
    prepared_image_path = prepare_image(path)

    client = vision.ImageAnnotatorClient()
    with open(prepared_image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if not response.full_text_annotation.pages:
        print("No text found")
        return "None"

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f"\nBlock confidence: {block.confidence}\n")
            for paragraph in block.paragraphs:
                print("Paragraph confidence: {}".format(paragraph.confidence))
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    print("Word text: {} (confidence: {})".format(word_text, word.confidence))
                    for symbol in word.symbols:
                        print("\tSymbol: {} (confidence: {})".format(symbol.text, symbol.confidence))

    if response.error.message:
        raise Exception("{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors".format(response.error.message))

# Example usage
detect_document('finalOutput/page_3/cell_108.png')
