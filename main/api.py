import boto3
from botocore.config import Config

# export AWS_DEFAULT_REGION=us-east-2

client = boto3.client('textract', aws_access_key_id='AKIASYEFUGCXXA7JYW5R',
                      aws_secret_access_key='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')


def apiResult(image):
    with open(image, 'rb') as file:

        img_test = file.read()

        bytes_test = bytearray(img_test)

    response = client.analyze_document(
        Document={'Bytes': bytes_test}, FeatureTypes=['TABLES'])
    if len(response.get('Blocks', [])) > 1:
        outputText = response['Blocks'][1].get('Text', None)
    else:
        outputText = None
    print(outputText)
    return outputText
