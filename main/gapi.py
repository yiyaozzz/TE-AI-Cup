import os
from google.cloud import vision
from PIL import Image, ExifTags, ImageEnhance


# Set Google API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pravin/Desktop/TE_Comp/TE-AI-Cup/main/service_account_token.json"


def prepare_image(image_path, min_width=1024, min_height=768):
    """
    Prepare the image by checking dimensions, adding padding if necessary, correcting orientation,
    and enhancing text appearance by making it bolder and blacker.
    """
    with Image.open(image_path) as img:
        # Correct orientation based on EXIF data
        try:
            exif = {ExifTags.TAGS[k]: v for k,
                    v in img._getexif().items() if k in ExifTags.TAGS}
            if 'Orientation' in exif:
                if exif['Orientation'] == 3:
                    img = img.rotate(180, expand=True)
                elif exif['Orientation'] == 6:
                    img = img.rotate(270, expand=True)
                elif exif['Orientation'] == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # Cases where the image doesn't have getexif or EXIF data is not relevant
            pass

        # Enhance the image to make text bolder and darker
        enhancer = ImageEnhance.Contrast(img)
        # Adjust the factor to get the desired contrast level
        img = enhancer.enhance(1.5)

        width, height = img.size
        if width < min_width or height < min_height:
            new_width = max(min_width, width)
            new_height = max(min_height, height)
            new_img = Image.new("RGB", (new_width, new_height), "white")
            new_img.paste(img, ((new_width - width) //
                          2, (new_height - height) // 2))
            new_img_path = "test/processed_image.png"
            new_img.save(new_img_path, "PNG")
            return new_img_path
        else:
            # Save the potentially re-oriented and enhanced image
            img_path = "test/processed_image.png"
            img.save(img_path, "PNG")
            return img_path


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
        print(full_text)

    return full_document_text if full_document_text else None


# Example usage
# apiResult('test/image.png')
