import os
from PIL import Image


def resize_and_pad(img, target_width, target_height, color=(255, 255, 255)):
    img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

    padded_img = Image.new("RGB", (target_width, target_height), color)

    x = (target_width - img.width) // 2
    y = (target_height - img.height) // 2

    padded_img.paste(img, (x, y))

    return padded_img


def process_images_in_folder(folder_path, target_width=640, target_height=360):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            with Image.open(filepath) as img:
                processed_img = resize_and_pad(
                    img, target_width, target_height)
                processed_img.save(filepath, 'PNG')
                print(f"Processed and saved: {filename}")


# !pip install boto3

# Access Key: AKIASYEFUGCXXA7JYW5R

# Secret Access Key: 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
# [13:27] Osunkwo, Sonny O (Unverified)
# image to read text from

# with open('text_doc.png', 'rb') as file:

#     img_test = file.read()

#     bytes_test = bytearray(img_test)

# response = client.analyze_document(Document={'Bytes': bytes_test},FeatureTypes = ['TABLES'])

# print(response)
# [13:27] Osunkwo, Sonny O (Unverified)
# # image to read text from

# with open('C:/Users/TE356362/Downloads/MicrosoftTeams-image (11).png', 'rb') as file:

#     img_test = file.read()

#     bytes_test = bytearray(img_test)

# response = client.analyze_document(Document={'Bytes': bytes_test},FeatureTypes = ['TABLES'])

# print(response)
