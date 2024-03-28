from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from keras.models import Model
import numpy as np
import cv2
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def pad_and_resize(image):
    desired_size = 224
    old_size = image.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    new_size = tuple(min(new_size[i], desired_size) for i in range(2))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    padding = (delta_w//2, delta_h//2, delta_w -
               (delta_w//2), delta_h-(delta_h//2))

    image = cv2.copyMakeBorder(
        image, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT, value=[0, 0, 0])

    if new_size != (desired_size, desired_size):
        image = cv2.resize(image, (desired_size, desired_size))

    return image


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_directory = 'datasets/train'
validation_directory = 'datasets/val'

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')


def build_vgg13(input_shape=(224, 224, 3), num_classes=10):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu',
               padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


num_classes = len(train_generator.class_indices)
model = build_vgg13(input_shape=(224, 224, 3), num_classes=num_classes)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10)

feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_model.predict(train_generator)

pca = PCA(n_components=50)
pca_features = pca.fit_transform(features)

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")


predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)


true_classes = validation_generator.classes


class_labels = list(validation_generator.class_indices.keys())


print(classification_report(true_classes,
      predicted_classes, target_names=class_labels))


overall_accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Overall Validation Accuracy: {overall_accuracy:.2f}")
