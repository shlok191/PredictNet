# Importing required files!

from tensorflow import keras
import tensorflow as tf
import numpy as np


def recurrent_encoder(state_size: int = 64, output_dims: int = 128, time_horizon: int = 10 , num_features: int = 7, verbose=True):
    
    """
        Defines an encoding model to produce a 128-dimensional embedding for a pedestrian's trajectory via LSTMs!

        Parameters:
            - state_size: The number of LSTM units in each layer.
            - output_dims: Determines the dimensions of the output vector.
            - time_horizon: The total timesteps (H) covered in an agent's past trajectory.
            - num_features: The number of features in each timestep.
            - verbose: Determines if model summary is printed.

        Returns:
            - A Keras model with two LSTM layers that outputs a 128-dimensional embedding vector.
    """

    # Defining our base model
    model = keras.models.Sequential()

    # Adding the first model layer with input tensor shape [Horizon_n x Features_n]
    model.add(
        keras.layers.LSTM(
            state_size, 
            input_shape=(time_horizon, num_features), 
            return_sequences=True
        )
    )

    # Adding the second layer with an inferred output tensor shape with a default state size of 128
    model.add(
        keras.layers.LSTM(
            state_size, 
            return_sequences=False
        )
    )

    # Finally, converts our hidden state encoding into desired output dimensions!
    model.add(keras.layers.Dense(output_dims))


    # Describe the model if requested for better user understanding
    if(verbose):
        print(model.summary())


    return model
