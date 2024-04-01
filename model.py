import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import sys

# Function to load and preprocess the image


def load_and_preprocess_image(img_path, img_side=28):
    # Load the image file
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    # Resize to model input size
    img = img.resize((img_side, img_side), Image.ANTIALIAS)
    img_array = np.array(img)
    # Reshape for model
    img_array = img_array.reshape(-1, img_side, img_side, 1)
    img_array = img_array.astype('float32') / 255.  # Normalize to [0, 1]
    return img_array

# Main function to predict the class of the image


def predict_image_class(img_path, model_path='mnist_cnn_model.h5'):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Load the trained model
    model = load_model(model_path)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class


# Using the script
if __name__ == '__main__':
    # Path to the image file from command line arguments
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'E.png'

    predicted_class = predict_image_class(img_path)
    print(f"Predicted Class: {predicted_class}")

# Note: Replace 'path/to/your/image.png' with the actual image path you want to predict.
