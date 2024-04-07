from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import pickle
from datetime import datetime

# Paths to your dataset directories
train_path = 'datasets/train'
valid_path = 'datasets/val'
test_path = 'datasets/test'

# Load your pre-trained model (Make sure the path is correct)
model_path = 'VGG16.h5'
model = load_model(model_path)

# ImageDataGenerator configuration


def configure_datagen():
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing_function,
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=False,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing_function,
        rescale=1./255
    )

    return train_datagen, test_datagen


def preprocess_image(img):
    img = preprocess_input(img)
    return img


def custom_preprocessing_function(img):
    img_processed = preprocess_image(img)
    return img_processed


def get_predictions(model, datagen, directory_path, batch_size=64):
    test_generator = datagen.flow_from_directory(
        directory_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_steps_per_epoch = np.math.ceil(
        test_generator.samples / test_generator.batch_size)
    predictions = model.predict_generator(
        test_generator, steps=test_steps_per_epoch)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_probabilities = np.max(predictions, axis=1)

    return predictions, predicted_class_indices, predicted_probabilities, test_generator.class_indices


def save_predictions(predictions, file_path='predictions.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = []

    data.append(predictions)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    train_datagen, test_datagen = configure_datagen()

    predictions, predicted_class_indices, predicted_probabilities, class_indices = get_predictions(
        model, test_datagen, test_path)
    save_predictions(predictions)

    # Inverting the class_indices dictionary to map indices back to class names
    index_to_class = {v: k for k, v in class_indices.items()}

    # Print the predicted class and probability for each image
    for i, predicted_class_index in enumerate(predicted_class_indices):
        class_name = index_to_class[predicted_class_index]
        probability = predicted_probabilities[i]
        print(
            f"Image {i}: Predicted class: {class_name} with probability {probability:.2f}")
    print(predictions)

    print("Predictions saved successfully.")

# from keras.models import load_model
# from keras.applications.resnet50 import preprocess_input
# import numpy as np
# import cv2
# import os
# from keras.preprocessing.image import ImageDataGenerator
# import pickle
# import os


# def process_images_in_directory(directory_path):
#     images = []
#     for filename in os.listdir(directory_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(directory_path, filename)
#             processed_img = preprocess_image(img_path)
#             images.append(processed_img)

#     return np.vstack(images)


# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise Exception(f"Image not found at {image_path}")

#     desired_size = 224
#     old_size = img.shape[:2]
#     ratio = float(desired_size)/max(old_size)
#     new_size = tuple([int(x*ratio) for x in old_size])
#     img = cv2.resize(img, (new_size[1], new_size[0]))
#     new_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
#     x_offset = (desired_size - new_size[1]) // 2
#     y_offset = (desired_size - new_size[0]) // 2

#     new_img[y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = img

#     new_img = preprocess_input(new_img.astype('float32'))
#     new_img = np.expand_dims(new_img, axis=0)

#     return new_img


# model_path = 'VGG16.h5'
# model = load_model(model_path)
# directory_path = 'datasets/test'
# # img = process_images_in_directory(directory_path)
# test_generator = ImageDataGenerator(rescale=1./255)
# test_data_generator = test_generator.flow_from_directory(directory_path,
#                                                          target_size=(
#                                                              224, 224),
#                                                          batch_size=64,
#                                                          shuffle=False,
#                                                          class_mode="categorical")
# test_steps_per_epoch = np.math.ceil(
#     test_data_generator.samples / test_data_generator.batch_size)
# predictions = model.predict_generator(
#     test_data_generator, steps=test_steps_per_epoch)


# # predictions = model.predict(img)
# predicted_class_indices = np.argmax(predictions, axis=1)
# predicted_probabilities = np.max(predictions, axis=1)

# labels_list = ['1', '2', '3', '4', '5', 'DISC', 'EH', 'EW', 'N_A']
# # {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, 'DISC': 5, 'EH': 6, 'EW': 7, 'N_A': 8}
# print("Predictions 2D Array:")
# print(predictions)  # This will print the entire 2D array of predictions

# for i, predicted_class_index in enumerate(predicted_class_indices):
#     print(
#         f"Image {i}: Predicted class: {labels_list[predicted_class_index]} with probability {predicted_probabilities[i]:.2f}"
#     )
