# Importing required libraries!

import tensorflow as tf
from tensorflow import keras
import numpy as np

def map_encoder(input_shape, output_shape):

    # Defining our base model
    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(
            filters=4, 
            kernel_size=(3, 3), 
            strides=(   1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.MaxPool2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=8, 
            kernel_size=(3, 3), 
            strides=(   1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.MaxPool2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=16, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=16, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.MaxPool2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.MaxPool2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation='relu'
        )
    )

    model.add(
        keras.layers.MaxPool2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        keras.layers.Dense(
            units=512,
            activation='relu'
        )
    )

    model.add(
        keras.layers.Dense(
            units=100,
        )
    )

    return model


