from tensorflow.python.keras.models import load_model
import numpy as np
import pandas as pd
from glob import glob
import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.python.keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential

train_path = 'Datasets/train'
valid_path = 'Datasets/test'
test_path = 'Datasets/val'

IMAGE_SIZE = [224, 224]  # Default image size for VGG16

folders = glob('Datasets/train/*')  # Get number of classes


# ImageDataGenerator can help perform augumentation, get more diverse train set.
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Through flow_from_directory - array of images that can be used for training.
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='categorical')

validation_set = validation_datagen.flow_from_directory('Datasets/val',
                                                        target_size=(224, 224),
                                                        batch_size=64,
                                                        class_mode='categorical')

test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# Create a VGG16 model, and removing the last layer that is classifying 1000 images. This will be replaced with images classes we have.
# Training with Imagenet weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

# Use this line for VGG19 network. Create a VGG19 model, and removing the last layer that is classifying 1000 images. This will be replaced with images classes we have.
# vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# This sets the base that the layers are not trainable. If we'd want to train the layers with custom data, these two lines can be ommitted.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)  # Output obtained on vgg16 is now flattened.
# We have 5 classes, and so, the prediction is being done on len(folders) - 5 classes
prediction = Dense(len(folders), activation='softmax')(x)

# Creating model object
model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(
    training_set, validation_data=validation_set, epochs=20, batch_size=32)

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


model.save('FlowerClassification_VGG16.h5')
