# prediction.py

import torch.nn as nn
import torch
from torchvision import transforms, models
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 7)
model_path = 'resnet_complete.pth'


model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Prediction function


def predict_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs.data, 1)
    # Return the index of the predicted class
    return predicted_idx.item()


image_path = 'datasets/val/Tally 2/Screenshot 2024-04-03 144849.jpg'
predicted_class_index = predict_image(image_path)


class_names = ['N/A', 'TALLY 1', 'TALLY 2', 'TALLY 3', 'TALLY 4',
               'TALLY 5', 'Word']
print(
    f'Predicted class for the input image is: {class_names[predicted_class_index]}')
