import boto3
from botocore.config import Config
from PIL import Image


# export AWS_DEFAULT_REGION=us-east-2
client = boto3.client('textract', aws_access_key_id='AKIASYEFUGCXXA7JYW5R',
                      aws_secret_access_key='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')


def increase_threshold(image_path):
    with Image.open(image_path) as img:
        gray_img = img.convert('L')
        threshold_value = 250
        threshold_img = gray_img.point(lambda p: p > threshold_value and 255)
        threshold_img.save("test/ok.png")


def aapiResult(image):
    increase_threshold(image)
    with open(image, 'rb') as file:

        img_test = file.read()

        bytes_test = bytearray(img_test)

    response = client.analyze_document(
        Document={'Bytes': bytes_test}, FeatureTypes=['TABLES'])
    if len(response.get('Blocks', [])) > 1:
        outputText = response['Blocks'][1].get('Text', None)
    else:
        outputText = None
        print("This image is not detected " + image)
    print(outputText)
    return outputText


aapiResult("nextjs/works/finalOutput_b2518793-8e06-4ec7-aa7a-954ee56bb933.pdf/page_16/row_3/column_4/24_Words.png")
# finalOutput/page_11/row_4/column_4/1_Number.jpg
