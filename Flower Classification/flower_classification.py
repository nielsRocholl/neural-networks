import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from preprocessing.image_preprocessing import image_input_pipeline
from tuning.hyper_parameter_tuning import create_hypermodel

# Supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get tf version and see if gpu is enabled
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def visualize_data(ds, class_names):
    """
    Plot 10 samples from the dataset
    """
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis('off')
    plt.show()


def model_builder(hp):
    model = keras.Sequential()
    model.add(Rescaling(1. / 255))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    # Tune the number of units in the Dense layer
    # Choose an optimal value between 32-512
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=units, activation='relu'))
    model.add(Dense(5))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def main():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname='flower_photos',
                                       untar=True)

    # # Create a dataset
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f'Found Classes: {class_names}')

    # Visualize data
    visualize_data(train_ds, class_names)

    batch, labels = next(iter(train_ds))
    print(f'Batch Shape: {batch.shape}\nLabels Shape: {labels.shape}')

    # Standardize data
    normalization_layer = Rescaling(1. / 255)

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Create model
    num_classes = len(class_names)
    model = Sequential([
        normalization_layer,
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=1)

    train_ds, val_ds = image_input_pipeline(data_dir=data_dir)

    print('Train model with data preprocessed by "image_input_pipeline":')
    model.fit(train_ds, validation_data=val_ds, epochs=1)

    # train hypermodel
    hypermodel, best_epoch = create_hypermodel(train_ds, val_ds, model_builder)
    hypermodel.fit(train_ds, validation_data=val_ds, epochs=best_epoch)


if __name__ == '__main__':
    main()
