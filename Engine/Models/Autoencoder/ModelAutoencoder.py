#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    import sys
    import numpy
    
    from typing import List
    from typing import Tuple
    from typing import Optional

    from Engine.Models.Autoencoder.VanillaDecoder import VanillaDecoder
    from Engine.Models.Autoencoder.VanillaEncoder import VanillaEncoder


except ImportError as error:
    print(error)
    sys.exit(-1)


DEFAULT_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_AUTOENCODER_ACTIVATION = "swish"
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_AUTOENCODER_INITIALIZER_MEAN = 0.0
DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION = 0.125


class AutoencoderModel(VanillaEncoder, VanillaDecoder):
    """
    AutoencoderModel

    This class implements an Autoencoder model by inheriting from the VanillaEncoder and VanillaDecoder classes.
    It constructs an autoencoder architecture by combining both an encoder and a decoder with customizable
    hyperparameters. The autoencoder is typically used for tasks such as dimensionality reduction, feature learning,
    and denoising.

    Attributes:
        @latent_dimension (int):
            The dimensionality of the latent space (encoding space).
        @output_shape (tuple):
            The desired shape of the output of the decoder.
        @activation_function (str):
            The activation function used throughout the encoder and decoder.
        @initializer_mean (float):
            The mean for weight initialization.
        @initializer_deviation (float):
            The standard deviation for weight initialization.
        @dropout_decay_encoder (float):
            The rate of decay for the dropout layer in the encoder.
        @dropout_decay_decoder (float):
            The rate of decay for the dropout layer in the decoder.
        @last_layer_activation (str):
            The activation function for the last layer in both encoder and decoder.
        @number_neurons_encoder (List[int]):
            A list specifying the number of neurons for each layer of the encoder.
        @number_neurons_decoder (List[int]):
            A list specifying the number of neurons for each layer of the decoder.
        @dataset_type (numpy.dtype):
            The data type for the dataset (default is numpy.float32).
        @number_samples_per_class (Optional[int]):
            The number of samples per class, used for dataset management (optional).

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension`
            - Invalid `activation_function`, `initializer_mean`, or `initializer_deviation`
            - Invalid `dropout_decay_encoder` or `dropout_decay_decoder`
            - Mismatched types for `number_neurons_encoder` or `number_neurons_decoder`
            - Invalid `last_layer_activation`

    Example:
        >>> autoencoder_model = AutoencoderModel(
        ...     latent_dimension=64,
        ...     output_shape=(28, 28, 1),
        ...     activation_function="ReLU",
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.01,
        ...     dropout_decay_encoder=0.3,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation="sigmoid",
        ...     number_neurons_encoder=[128, 64, 32],
        ...     number_neurons_decoder=[32, 64, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class=1000
        ... )
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_AUTOENCODER_LATENT_DIMENSION,
                 output_shape: Tuple[int, ...] = (128,),
                 activation_function: str = DEFAULT_AUTOENCODER_ACTIVATION,
                 initializer_mean: float = DEFAULT_AUTOENCODER_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION,
                 dropout_decay_encoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_decoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                 last_layer_activation: str = DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER,
                 number_neurons_encoder=None,
                 number_neurons_decoder=None,
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[int] = None):
        """
        Initializes the AutoencoderModel by setting up both the encoder and decoder with the provided parameters.

        Args:
            latent_dimension (int): The dimensionality of the latent space (encoding space).
            output_shape (tuple): The desired shape of the output of the decoder.
            activation_function (str): The activation function to be used throughout the encoder and decoder.
            initializer_mean (float): The mean for weight initialization.
            initializer_deviation (float): The standard deviation for weight initialization.
            dropout_decay_encoder (float): The rate of decay for the dropout layer in the encoder.
            dropout_decay_decoder (float): The rate of decay for the dropout layer in the decoder.
            last_layer_activation (str): The activation function to be used for the last layer in both encoder and decoder.
            number_neurons_encoder (list): A list specifying the number of neurons for each layer of the encoder.
            number_neurons_decoder (list): A list specifying the number of neurons for each layer of the decoder.
            dataset_type (numpy.dtype): The data type for the dataset (default is numpy.float32).
            number_samples_per_class (int): The number of samples per class, used for dataset management (optional).

        Raises:
            ValueError:
                If `latent_dimension` or `activation_function` is invalid.
                If the values of `initializer_mean` or `initializer_deviation` are not numerical.
                If `dropout_decay_encoder` or `dropout_decay_decoder` are not within the range [0, 1].
                If `number_neurons_encoder` or `number_neurons_decoder` is not a list of integers.
                If `last_layer_activation` is invalid.
        """
        if number_neurons_decoder is None:
            number_neurons_decoder = DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER

        if number_neurons_encoder is None:
            number_neurons_encoder = DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")
    
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")
    
        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or integer.")
    
        if not isinstance(initializer_deviation, (float, int)):
            raise ValueError("initializer_deviation must be a float or integer.")
    
        if not isinstance(dropout_decay_encoder, (float, int)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float or integer between 0 and 1.")
    
        if not isinstance(dropout_decay_decoder, (float, int)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError("dropout_decay_decoder must be a float or integer between 0 and 1.")
    
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")
    
        if not isinstance(number_neurons_encoder, list) or not all(isinstance(x, int) for x in number_neurons_encoder):
            raise ValueError("number_neurons_encoder must be a list of integers.")
    
        if not isinstance(number_neurons_decoder, list) or not all(isinstance(x, int) for x in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of integers.")

        VanillaDecoder.__init__(self,
                                latent_dimension,
                                output_shape,
                                activation_function,
                                initializer_mean,
                                initializer_deviation,
                                dropout_decay_decoder,
                                last_layer_activation,
                                number_neurons_decoder,
                                dataset_type,
                                number_samples_per_class)

        VanillaEncoder.__init__(self,
                                latent_dimension,
                                output_shape,
                                activation_function,
                                initializer_mean,
                                initializer_deviation,
                                dropout_decay_encoder,
                                last_layer_activation,
                                number_neurons_encoder,
                                dataset_type,
                                number_samples_per_class)

    def get_dense_encoder_model(self):
        """
        Returns the encoder model.

        Returns:
            tensorflow.keras.Model: The model representing the encoder.
        """
        return self._encoder_model

    def get_dense_decoder_model(self):
        """
        Returns the decoder model.

        Returns:
            tensorflow.keras.Model: The model representing the decoder.
        """
        return self._decoder_model

    def latent_dimension(self, latent_dimension: int) -> None:
        """
        Sets the latent dimension for both the encoder and decoder.

        Args:
            latent_dimension (int): The dimensionality of the latent space.
        """
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        self._encoder_latent_dimension = latent_dimension
        self._decoder_latent_dimension = latent_dimension

    def output_shape(self, output_shape: Tuple[int, ...]) -> None:
        """
        Sets the output shape for both the encoder and decoder.

        Args:
            output_shape (tuple): The shape of the output to be generated by both the encoder and decoder.
        """
        if not isinstance(output_shape, tuple) or not all(isinstance(x, int) for x in output_shape):
            raise ValueError("output_shape must be a tuple of integers.")

        self._encoder_output_shape = output_shape
        self._decoder_output_shape = output_shape

    def activation_function(self, activation_function: str) -> None:
        """
        Sets the activation function for both the encoder and decoder.

        Args:
            activation_function (str): The activation function to be applied throughout the encoder and decoder.
        """
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        self._encoder_activation_function = activation_function
        self._decoder_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation: str) -> None:
        """
        Sets the activation function for the last layer of both the encoder and decoder.

        Args:
            last_layer_activation (str): The activation function to be used in the last layer of both the encoder and decoder.
        """
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        self._encoder_last_layer_activation = last_layer_activation
        self._decoder_last_layer_activation = last_layer_activation
