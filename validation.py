from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
import os


def process_images_in_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            processed_img = preprocess_image(img_path)
            images.append(processed_img)

    if not images:
        raise Exception("No images found in the directory")

    return np.vstack(images)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Image not found at {image_path}")

    desired_size = 224
    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    new_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    x_offset = (desired_size - new_size[1]) // 2
    y_offset = (desired_size - new_size[0]) // 2

    new_img[y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = img

    new_img = preprocess_input(new_img.astype('float32'))
    new_img = np.expand_dims(new_img, axis=0)

    return new_img


model_path = 'VGG16.h5'
model = load_model(model_path)
directory_path = 'datasets/val/EW'
image_path = process_images_in_directory(directory_path)


img = image_path

predictions = model.predict(img)
predicted_class_index = np.argmax(predictions, axis=1)
predicted_probability = np.max(predictions, axis=1)

labels_list = ['DISC', 'EW']

for i, predicted_class_index in enumerate(predicted_class_index):
    print(
        f"Image {i}: Predicted class: {labels_list[predicted_class_index]} with probability {predicted_probability[i]:.2f}"
    )
