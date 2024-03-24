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

train_path = 'datasets/train'
valid_path = 'datasets/test'
test_path = 'datasets/val'

train_labels = sorted(os.listdir('datasets/train'))
val_labels = sorted(os.listdir('datasets/val'))

IMAGE_SIZE = [224, 224]  # size for VGG16

folders = glob('datasets/train/*')
labels_list = list(train_labels)
# EW ||||


def preprocess_image(img):
    desired_size = 224
    old_size = img.shape[:2]  # old_size in (height, width)

    # scale the image accordingly
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # Resize the image to new_size
    img = cv2.resize(img, (new_size[1], new_size[0]))

    # Create a new image of desired size and black color (for padding)
    new_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    x_offset = (desired_size - new_size[1]) // 2  # Center
    y_offset = (desired_size - new_size[0]) // 2

    new_img[y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = img
    # print("Final image size:", new_img.shape)

    return new_img

# Define a custom ImageDataGenerator preprocessing function


def custom_preprocessing_function(img):
    # Convert the image to an array format
    img_array = image.img_to_array(img)
    # Apply the preprocess and padding/resizing function
    img_processed = preprocess_image(img_array)
    # Further preprocess the image for the model
    img_processed = preprocess_input(img_processed)
    return img_processed

# List for labels
# Keras model
# shear and zoom
# Add more variations to imadedatagen
# number of itts
#


train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,  # Changed to True to allow flips
    rotation_range=20,  # Rotate images slightly
    width_shift_range=0.2,  # Shift the image widthwise
    height_shift_range=0.2  # Shift the image heightwise
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function
)

test_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function
)

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('datasets/train',
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='categorical')

validation_set = validation_datagen.flow_from_directory('datasets/val',
                                                        target_size=(224, 224),
                                                        batch_size=64,
                                                        class_mode='categorical')

test_set = test_datagen.flow_from_directory('datasets/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

print("Training set class indices:", training_set.class_indices)
print("Validation set class indices:", validation_set.class_indices)
print("Test set class indices:", test_set.class_indices)


# Training with Imagenet weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

# This sets the base that the layers are not trainable.
for layer in vgg.layers:
    layer.trainable = False


# Custom top layers
x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(len(folders), activation='softmax')(x)

# Creating model object
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model with a custom learning rate
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['accuracy'])
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)

class_weight_dict = dict(enumerate(class_weights))


# # Compile the model
# vgg.compile(loss='categorical_crossentropy',
#             optimizer='adam', metrics=['accuracy'])
# classes = np.unique(training_set.classes)
# class_weights = compute_class_weight(
#     'balanced', classes=classes, y=training_set.classes)
# class_weight_dict = dict(zip(classes, class_weights))
# Use this in model.fit
history = model.fit(training_set,
                    validation_data=validation_set,
                    epochs=11,
                    batch_size=32,
                    class_weight=class_weight_dict)


model.summary()


# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


model.save('VGG16.h5')
