from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import math


def process_images_in_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            processed_img = preprocess_image(img_path)
            images.append(processed_img)

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
    # new_img = new_img ./255
    new_w = int(img.shape[1] * .255)
    new_h = int(img.shape[0] * .255)
    dime = (new_w, new_h)
    # final_img = cv2.resize(new_img, dime, interpolation=cv2.INTER_AREA)
    new_img = np.expand_dims(new_img, axis=0)

    return new_img


model_path = 'VGG16.h5'
model = load_model(model_path)
directory_path = 'datasets/test'
# img = process_images_in_directory(directory_path)
test_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = test_generator.flow_from_directory(directory_path,
                                                         target_size=(
                                                             224, 224),
                                                         batch_size=64,
                                                         shuffle=False,
                                                         class_mode="categorical")
test_steps_per_epoch = np.math.ceil(
    test_data_generator.samples / test_data_generator.batch_size)
predictions = model.predict_generator(
    test_data_generator, steps=test_steps_per_epoch)


# predictions = model.predict(img)
predicted_class_indices = np.argmax(predictions, axis=1)
predicted_probabilities = np.max(predictions, axis=1)

labels_list = ['1', '2', '3', '4', '5', 'DISC', 'EH', 'EW', 'N_A']
# {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, 'DISC': 5, 'EH': 6, 'EW': 7, 'N_A': 8}
print("Predictions 2D Array:")
print(predictions)  # This will print the entire 2D array of predictions

for i, predicted_class_index in enumerate(predicted_class_indices):
    print(
        f"Image {i}: Predicted class: {labels_list[predicted_class_index]} with probability {predicted_probabilities[i]:.2f}"
    )
