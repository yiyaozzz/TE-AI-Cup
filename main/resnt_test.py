import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Model setup
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model_path = 'resnet_tally.pth'

# Load trained model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


def preprocess_image(image):
    image = Image.fromarray(image).convert('RGB')
    image_tensor = transform(image).unsqueeze(
        0)
    return image_tensor

# Prediction function


def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs.data, 1)
    return predicted_idx.item()


def white_border(img, border_size):
    img[:border_size, :] = 255  # Top border
    img[-border_size:, :] = 255  # Bottom border
    img[:, :border_size] = 255  # Left border
    img[:, -border_size:] = 255  # Right border
    return img


# Image path
# image_path = 'runs/detect/track18/crops/Words-and-tallys/cell_45.jpg'


def resnetPred(image_path):
    img = cv2.imread(image_path)  # Load image
    if img is None:
        print("Failed to load image")
    else:
        img = white_border(img, 1)

        image_tensor = preprocess_image(img)
        predicted_class_index = predict_image(image_tensor)

        # Class names
        # 'N/A', 'Scratch',
        class_names = ['TALLY 1', 'TALLY 2', 'TALLY 3', 'TALLY 4', 'TALLY 5']
        print(
            f'Predicted class for the input image is: {class_names[predicted_class_index]}')
        return class_names[predicted_class_index]
