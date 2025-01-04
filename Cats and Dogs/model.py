import tensorflow as tf
import keras
from keras import layers

import matplotlib.pyplot as plt
import numpy as np
import random
import os

train, val = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    label_mode="binary",
    batch_size=128,
    image_size=(180, 180),
    validation_split=0.2,
    subset="both",
    seed=18051809
)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


inputs = keras.Input(shape=(180,180,3))
x = data_augmentation(inputs)
x = layers.Rescaling(scale=1./255)(inputs)
x = layers.Conv2D(64, 3, padding="same", strides=2)(x) # (90,90,128)
x = layers.Activation("relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x) # (45,45,128)
x = layers.Conv2D(128, 3, padding="same", strides=2)(x) # (15, 15, 256)
x = layers.Activation("relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(pool_size=(2,2))(x) # (5,5,256)
x = layers.Conv2D(256, 4, padding="valid", strides=2)(x) # (15, 15, 256)
x = layers.Activation("relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(pool_size=(2,2))(x) # (5,5,256)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(train, validation_data=val, callbacks=[keras.callbacks.ModelCheckpoint("save_at_{epoch}_v2.keras")], epochs=25)