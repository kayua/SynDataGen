#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/10/29'
__credits__ = ['Kayuã Oleques']

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

import sys
import numpy
from typing import Any, Dict, Optional, Tuple, Union, List

# Detect available framework
FRAMEWORK = None
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal
    FRAMEWORK = 'tensorflow'
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    if FRAMEWORK is None:
        FRAMEWORK = 'pytorch'
except ImportError:
    pass

if FRAMEWORK is None:
    print("Error: Neither TensorFlow nor PyTorch is installed.")
    sys.exit(-1)


class VanillaEncoder:
    """
    VanillaEncoder (Framework Agnostic)

    A class representing a Vanilla Encoder model for deep learning applications. The encoder
    is designed to process inputs and labels, apply a series of dense layers with activations
    and dropout, and output a latent representation of the input data.

    This implementation automatically detects and uses either TensorFlow or PyTorch.

    Attributes:
        latent_dimension (int):
            The dimensionality of the latent space that the model will output.
        output_shape (tuple):
            The desired output shape of the encoder.
        activation_function (str):
            The activation function applied to each layer (e.g., 'relu', 'leaky_relu').
        last_layer_activation (str):
            The activation function applied to the final output layer.
        dropout_decay_encoder (float):
            The rate of dropout applied during encoding (must be between 0 and 1).
        number_neurons_encoder (list):
            A list specifying the number of neurons in each layer of the encoder network.
        dataset_type (dtype):
            The data type of the input dataset, default is numpy.float32.
        initializer_mean (float):
            The mean for the normal distribution used to initialize the weights.
        initializer_deviation (float):
            The standard deviation for weight initialization.
        number_samples_per_class (Optional[dict]):
            Optional dictionary containing the number of samples per class.

    Example:
        >>> encoder = VanillaEncoder(
        ...     latent_dimension=128,
        ...     output_shape=(64, 64, 1),
        ...     activation_function='relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = encoder.get_encoder(input_shape=784)
    """

    def __new__(cls, *args, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Returns:
            VanillaEncoderTF or VanillaEncoderPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return object.__new__(VanillaEncoderTF)
        elif FRAMEWORK == 'pytorch':
            return object.__new__(VanillaEncoderPyTorch)


class VanillaEncoderTF:
    """TensorFlow implementation of VanillaEncoder."""

    def __init__(self, 
                 latent_dimension: int, 
                 output_shape: Union[tuple, int], 
                 activation_function: str, 
                 initializer_mean: float,
                 initializer_deviation: float, 
                 dropout_decay_encoder: float, 
                 last_layer_activation: str,
                 number_neurons_encoder: List[int], 
                 dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None):
        
        self._validate_parameters(latent_dimension, initializer_mean, initializer_deviation,
                                 dropout_decay_encoder, number_neurons_encoder, number_samples_per_class)

        self._encoder_latent_dimension = latent_dimension
        self._encoder_output_shape = output_shape
        self._encoder_activation_function = activation_function.lower()
        self._encoder_last_layer_activation = last_layer_activation.lower()
        self._encoder_dropout_decay_rate_encoder = dropout_decay_encoder
        self._encoder_dataset_type = dataset_type
        self._encoder_initializer_mean = initializer_mean
        self._encoder_initializer_deviation = initializer_deviation
        self._encoder_number_neurons_encoder = number_neurons_encoder
        self._encoder_number_samples_per_class = number_samples_per_class

    def _validate_parameters(self, latent_dimension, initializer_mean, initializer_deviation,
                           dropout_decay_encoder, number_neurons_encoder, number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a number.")
        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError("initializer_deviation must be a number.")
        if not isinstance(dropout_decay_encoder, (int, float)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float between 0 and 1.")
        if not isinstance(number_neurons_encoder, list) or len(number_neurons_encoder) == 0:
            raise ValueError("number_neurons_encoder must be a non-empty list.")
        for neurons in number_neurons_encoder:
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError("Each element in number_neurons_encoder must be a positive integer.")
        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary.")

    def get_encoder(self, input_shape: Union[int, tuple]) -> Model:
        """
        Creates and returns the encoder model (TensorFlow).

        Args:
            input_shape (int or tuple): The shape of the input data.

        Returns:
            keras.Model: The encoder model.
        """
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("`number_samples_per_class` must include a 'number_classes' key.")

        # Handle input_shape as tuple or int
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0] if len(input_shape) == 1 else input_shape
        
        initialization = RandomNormal(mean=self._encoder_initializer_mean, 
                                     stddev=self._encoder_initializer_deviation)

        # Define input layers
        neural_model_inputs = Input(shape=(input_shape,), dtype=self._encoder_dataset_type, name="first_input")
        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                          dtype=self._encoder_dataset_type, name="second_input")

        # Concatenate and build encoder
        x = Concatenate()([neural_model_inputs, label_input])
        x = Dense(self._encoder_number_neurons_encoder[0], kernel_initializer=initialization)(x)
        x = Dropout(self._encoder_dropout_decay_rate_encoder)(x)
        x = tf.keras.layers.Activation(self._encoder_activation_function)(x)

        for number_neurons in self._encoder_number_neurons_encoder[1:]:
            x = Dense(number_neurons, kernel_initializer=initialization)(x)
            x = Dropout(self._encoder_dropout_decay_rate_encoder)(x)
            x = tf.keras.layers.Activation(self._encoder_activation_function)(x)

        # Map to latent space
        x = Dense(self._encoder_latent_dimension, kernel_initializer=initialization)(x)
        encoder_output = tf.keras.layers.Activation(self._encoder_last_layer_activation)(x)

        return Model([neural_model_inputs, label_input], [encoder_output, label_input], name="Encoder")

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """Gets the dropout decay rate."""
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> List[int]:
        """Gets the number of neurons for each encoder layer."""
        return self._encoder_number_neurons_encoder

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_encoder: float) -> None:
        """Sets the dropout decay rate."""
        if not (0 <= dropout_decay_rate_encoder <= 1):
            raise ValueError("dropout_decay_rate_encoder must be a float between 0 and 1.")
        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_encoder


class VanillaEncoderPyTorch:
    """PyTorch implementation of VanillaEncoder."""

    def __init__(self, 
                 latent_dimension: int, 
                 output_shape: Union[tuple, int], 
                 activation_function: str, 
                 initializer_mean: float,
                 initializer_deviation: float, 
                 dropout_decay_encoder: float, 
                 last_layer_activation: str,
                 number_neurons_encoder: List[int], 
                 dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None):
        
        self._validate_parameters(latent_dimension, initializer_mean, initializer_deviation,
                                 dropout_decay_encoder, number_neurons_encoder, number_samples_per_class)

        self._encoder_latent_dimension = latent_dimension
        self._encoder_output_shape = output_shape
        self._encoder_activation_function = activation_function.lower()
        self._encoder_last_layer_activation = last_layer_activation.lower()
        self._encoder_dropout_decay_rate_encoder = dropout_decay_encoder
        self._encoder_dataset_type = dataset_type
        self._encoder_initializer_mean = initializer_mean
        self._encoder_initializer_deviation = initializer_deviation
        self._encoder_number_neurons_encoder = number_neurons_encoder
        self._encoder_number_samples_per_class = number_samples_per_class

    def _validate_parameters(self, latent_dimension, initializer_mean, initializer_deviation,
                           dropout_decay_encoder, number_neurons_encoder, number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a number.")
        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError("initializer_deviation must be a number.")
        if not isinstance(dropout_decay_encoder, (int, float)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float between 0 and 1.")
        if not isinstance(number_neurons_encoder, list) or len(number_neurons_encoder) == 0:
            raise ValueError("number_neurons_encoder must be a non-empty list.")
        for neurons in number_neurons_encoder:
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError("Each element in number_neurons_encoder must be a positive integer.")
        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary.")

    def _get_activation(self, activation_name: str):
        """Returns the appropriate PyTorch activation function."""
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activation_map.get(activation_name, nn.ReLU())

    def _initialize_weights(self, module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=self._encoder_initializer_mean,
                          std=self._encoder_initializer_deviation)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_encoder(self, input_shape: Union[int, tuple]):
        """
        Creates and returns the encoder model (PyTorch).

        Args:
            input_shape (int or tuple): The shape of the input data.

        Returns:
            nn.Module: The encoder model.
        """
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("`number_samples_per_class` must include a 'number_classes' key.")

        # Handle input_shape as tuple or int
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0] if len(input_shape) == 1 else input_shape

        class ConditionalEncoder(nn.Module):
            def __init__(self, input_dim, n_classes, layer_sizes, latent_dim, 
                        dropout_rate, activation_fn, last_activation_fn, init_fn):
                super(ConditionalEncoder, self).__init__()
                
                layers = []
                in_features = input_dim + n_classes
                
                # First layer
                layers.append(nn.Linear(in_features, layer_sizes[0]))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(activation_fn)
                
                # Hidden layers
                for layer_size in layer_sizes[1:]:
                    layers.append(nn.Linear(layer_sizes[layers.index(layer_size) // 3 - 1], layer_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(activation_fn)
                
                # Latent layer
                last_layer_size = layer_sizes[-1]
                layers.append(nn.Linear(last_layer_size, latent_dim))
                layers.append(last_activation_fn)
                
                self.model = nn.Sequential(*layers)
                self.apply(init_fn)
            
            def forward(self, data_input, label_input):
                x = torch.cat([data_input, label_input], dim=1)
                encoded = self.model(x)
                return encoded, label_input

        # Build properly with correct layer connections
        class ConditionalEncoderFixed(nn.Module):
            def __init__(self, input_dim, n_classes, layer_sizes, latent_dim, 
                        dropout_rate, activation_fn, last_activation_fn, init_fn):
                super(ConditionalEncoderFixed, self).__init__()
                
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = nn.ModuleList()
                
                in_features = input_dim + n_classes
                
                # Build layers
                for layer_size in layer_sizes:
                    self.layers.append(nn.Linear(in_features, layer_size))
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    self.activations.append(activation_fn)
                    in_features = layer_size
                
                # Latent layer
                self.latent_layer = nn.Linear(in_features, latent_dim)
                self.last_activation = last_activation_fn
                
                self.apply(init_fn)
            
            def forward(self, data_input, label_input):
                x = torch.cat([data_input, label_input], dim=1)
                
                for layer, dropout, activation in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = activation(x)
                
                x = self.latent_layer(x)
                encoded = self.last_activation(x)
                
                return encoded, label_input

        encoder = ConditionalEncoderFixed(
            input_shape,
            self._encoder_number_samples_per_class["number_classes"],
            self._encoder_number_neurons_encoder,
            self._encoder_latent_dimension,
            self._encoder_dropout_decay_rate_encoder,
            self._get_activation(self._encoder_activation_function),
            self._get_activation(self._encoder_last_layer_activation),
            self._initialize_weights
        )

        return encoder

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """Gets the dropout decay rate."""
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> List[int]:
        """Gets the number of neurons for each encoder layer."""
        return self._encoder_number_neurons_encoder

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_encoder: float) -> None:
        """Sets the dropout decay rate."""
        if not (0 <= dropout_decay_rate_encoder <= 1):
            raise ValueError("dropout_decay_rate_encoder must be a float between 0 and 1.")
        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_encoder


def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
