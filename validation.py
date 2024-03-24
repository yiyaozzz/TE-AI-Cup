from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

# Path to your saved model
model_path = 'VGG16.h5'
# Load the trained model
model = load_model(model_path)

# Function to preprocess the image


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # Resize the image to match model's expected input shape
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Scale pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Replace 'path_to_your_image.jpg' with the path to the image you want to predict
image_path = 'datasets/test/DISC/disc.png'
img = preprocess_image(image_path)

# Make a prediction
predictions = model.predict(img)

# Assuming your model outputs a softmax vector
predicted_class = np.argmax(predictions, axis=1)
predicted_probability = np.max(predictions)

# If you have the class labels as a list (from earlier when you printed class indices)
# example class names, replace with your actual class names
labels_list = ['DISC', 'EW']

# Print the prediction
print(
    f"Predicted class: {labels_list[predicted_class[0]]} with probability {predicted_probability:.2f}")
