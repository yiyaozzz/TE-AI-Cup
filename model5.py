from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import load_model
import numpy as np
import pandas as pd
from glob import glob
import keras
import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import cv2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

from keras import backend as K
# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)

# Path to datasets
train_path = 'datasets/train'
valid_path = 'datasets/test'
test_path = 'datasets/val'

# Assuming folders represent classes
folders = glob('datasets/train/*')
num_classes = len(folders)  # Number of classes

# Model definition


def build_model(input_shape=(28, 28, 1), num_classes=num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = build_model()

# Custom preprocessing function


def custom_preprocessing_function(img):
    # Check if the image has 3 channels (BGR format)
    if img.shape[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img  # If the image is already grayscale, use it directly

    img_resized = cv2.resize(img_gray, (28, 28))

    # Ensure the image is in the correct shape (28, 28, 1) for the CNN
    return img_resized.reshape(28, 28, 1) / 255.0


# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function)

# Data loading
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

val_set = val_datagen.flow_from_directory(
    valid_path,
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)
class_weight_dict = dict(enumerate(class_weights))

# Model training
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=20,
    validation_data=val_set,
    validation_steps=len(val_set),
    class_weight=class_weight_dict
)

# Plotting training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.show()

# Save the model
model.save('my_cnn_model.h5')
