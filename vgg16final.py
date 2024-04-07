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

# use vgg13 and remove the col 32, 32, 1
# vgg13, pca, classifier
train_path = 'datasets/train'
valid_path = 'datasets/test'
test_path = 'datasets/val'

train_labels = sorted(os.listdir('datasets/train'))
val_labels = sorted(os.listdir('datasets/val'))

IMAGE_SIZE = [224, 224]  # size for VGG16

folders = glob('datasets/train/*')
labels_list = list(train_labels)
# EW ||||

# border pixel value


i = 0


def preprocess_image(img):
    desired_size = 224
    old_size = img.shape[:2]  # old_size in (height, width)

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    # kern = np.ones((5,5))

    new_img = np.ones((desired_size, desired_size, 3), dtype=np.uint8)*255
    x_offset = (desired_size - new_size[1]) // 2  # Center
    y_offset = (desired_size - new_size[0]) // 2

    new_img[y_offset:y_offset+new_size[0], x_offset:x_offset +
            new_size[1]] = img
    global i
    new_file = 'temp' + str(i) + '.jpg'
    i = i+1
    save_path = os.path.join('temp', new_file)

    # cv2.imwrite(save_path, new_img)
    return new_img


def custom_preprocessing_function(img):
    img_array = image.img_to_array(img)
    img_processed = preprocess_image(img_array)
    img_processed = preprocess_input(img_processed)
    return img_processed

# List for labels
# Keras model
# shear and zoom
# Add more variations to imadedatagen
# number of itts
#
# Define a function to visualize the original and augmented images

# Noise add noise
# Change light intensity


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

validation_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing_function,
    rescale=1./255
)


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


vgg = VGG16(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

for layer in vgg.layers:
    if 'block5' in layer.name:
        layer.trainable = True
    elif 'block4' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(len(folders), activation='softmax')(x)


model = Model(inputs=vgg.input, outputs=prediction)
# METRIC


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# METRIC CLOSE
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['accuracy'])
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)

class_weight_dict = dict(enumerate(class_weights))

history = model.fit(training_set,
                    validation_data=validation_set,
                    epochs=15,
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
