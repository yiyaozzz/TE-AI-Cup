from keras.models import load_model, Model
import numpy as np
import pandas as pd
from glob import glob
import keras
import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import cv2
from sklearn.utils.class_weight import compute_class_weight
import os
from keras.optimizers import Adam

from keras import backend as K

train_path = 'datasets/train'
valid_path = 'datasets/test'
test_path = 'datasets/val'

IMAGE_SIZE = [224, 224]  # Standard size for ResNet50

folders = glob('datasets/train/*')


def preprocess_image(img):
    desired_size = 224
    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    new_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    x_offset = (desired_size - new_size[1]) // 2
    y_offset = (desired_size - new_size[0]) // 2
    new_img[y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = img
    return new_img


def custom_preprocessing_function(img):
    img_array = image.img_to_array(img)
    img_processed = preprocess_image(img_array)
    img_processed = preprocess_input(img_processed)
    return img_processed


train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    rescale=1./255
)

training_set = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=64, class_mode='categorical')
validation_set = validation_datagen.flow_from_directory(
    valid_path, target_size=(224, 224), batch_size=64, class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_path, target_size=(
    224, 224), batch_size=32, class_mode='categorical')

resnet = ResNet50(input_shape=IMAGE_SIZE +
                  [3], weights='imagenet', include_top=False)

for layer in resnet.layers:
    layer.trainable = False
# for layer in resnet.layers[:-1]:
#     layer.trainable = True

x = Flatten()(resnet.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=prediction)

model.compile(loss='categorical_crossentropy', optimizer=Adam(
    learning_rate=1e-5), metrics=['accuracy'])

class_weights = compute_class_weight('balanced', classes=np.unique(
    training_set.classes), y=training_set.classes)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(training_set, validation_data=validation_set,
                    epochs=30, batch_size=32, class_weight=class_weight_dict)

model.summary()
model.save('ResNet50.h5')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
