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
from typing import List, Dict, Optional, Union

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


class VanillaDecoder:
    """
    VanillaDecoder (Framework Agnostic)

    A class representing a conditional decoder model with support for customized dense layers,
    activation functions, dropout, and label-conditioned input. The decoder is designed to process
    a latent representation and output the desired shape.

    This implementation automatically detects and uses either TensorFlow or PyTorch.

    Attributes:
        latent_dimension (int):
            The dimensionality of the latent space input.
        output_shape (int):
            The dimensionality of the output layer.
        activation_function (str):
            The activation function applied to each layer (e.g., 'relu', 'leaky_relu').
        last_layer_activation (str):
            The activation function applied to the final output layer.
        dropout_decay_decoder (float):
            The rate of dropout applied during decoding (must be between 0 and 1).
        number_neurons_decoder (list):
            A list specifying the number of neurons in each layer of the decoder network.
        dataset_type (dtype):
            The data type of the input dataset, default is numpy.float32.
        initializer_mean (float):
            The mean for weight initialization.
        initializer_deviation (float):
            The standard deviation for weight initialization.
        number_samples_per_class (Optional[dict]):
            Optional dictionary containing the number of classes for label input.

    Example:
        >>> decoder = VanillaDecoder(
        ...     latent_dimension=128,
        ...     output_shape=784,
        ...     activation_function='relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[512, 256, 128],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = decoder.get_decoder(output_shape=784)
    """

    def __new__(cls, *args, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Returns:
            VanillaDecoderTF or VanillaDecoderPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return object.__new__(VanillaDecoderTF)
        elif FRAMEWORK == 'pytorch':
            return object.__new__(VanillaDecoderPyTorch)


class VanillaDecoderTF:
    """TensorFlow implementation of VanillaDecoder."""

    def __init__(self, 
                 latent_dimension: int, 
                 output_shape: int, 
                 activation_function: str, 
                 initializer_mean: float,
                 initializer_deviation: float, 
                 dropout_decay_decoder: float, 
                 last_layer_activation: str,
                 number_neurons_decoder: List[int], 
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[Dict] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function, initializer_mean,
                                 initializer_deviation, dropout_decay_decoder, last_layer_activation,
                                 number_neurons_decoder, dataset_type, number_samples_per_class)

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_activation_function = activation_function.lower()
        self._decoder_last_layer_activation = last_layer_activation.lower()
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class

    def _validate_parameters(self, latent_dimension, output_shape, activation_function, initializer_mean,
                           initializer_deviation, dropout_decay_decoder, last_layer_activation,
                           number_neurons_decoder, dataset_type, number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid latent_dimension: {latent_dimension}. Must be a positive integer.")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid output_shape: {output_shape}. Must be a positive integer.")
        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid activation_function: {activation_function}. Must be a string.")
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid initializer_mean: {initializer_mean}. Must be a number.")
        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError(f"Invalid initializer_deviation: {initializer_deviation}. Must be a number.")
        if not isinstance(dropout_decay_decoder, (int, float)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError(f"Invalid dropout_decay_decoder: {dropout_decay_decoder}. Must be between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid last_layer_activation: {last_layer_activation}. Must be a string.")
        if not isinstance(number_neurons_decoder, list) or not all(isinstance(x, int) and x > 0 for x in number_neurons_decoder):
            raise ValueError(f"Invalid number_neurons_decoder: {number_neurons_decoder}. Must be a list of positive integers.")
        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be a valid type.")
        if number_samples_per_class is not None and (not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(f"Invalid number_samples_per_class: {number_samples_per_class}. Must be a dict with 'number_classes'.")

    def get_decoder(self, output_shape: int) -> Model:
        """
        Constructs and returns the decoder model (TensorFlow).

        Args:
            output_shape (int): The output dimensionality of the decoder.

        Returns:
            keras.Model: The constructed decoder model.
        """
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid output_shape: {output_shape}. Must be a positive integer.")

        initialization = RandomNormal(mean=self._decoder_initializer_mean, 
                                     stddev=self._decoder_initializer_deviation)

        # Define input layers
        neural_model_inputs = Input(shape=(self._decoder_latent_dimension,), 
                                   dtype=self._decoder_dataset_type)
        label_input = Input(shape=(self._decoder_number_samples_per_class["number_classes"],),
                          dtype=self._decoder_dataset_type)

        # Concatenate and build decoder
        x = Concatenate()([neural_model_inputs, label_input])
        x = Dense(self._decoder_number_neurons_decoder[0], kernel_initializer=initialization)(x)
        x = Dropout(self._decoder_dropout_decay_rate_decoder)(x)
        x = tf.keras.layers.Activation(self._decoder_activation_function)(x)

        for number_filters in self._decoder_number_neurons_decoder[1:]:
            x = Dense(number_filters, kernel_initializer=initialization)(x)
            x = Dropout(self._decoder_dropout_decay_rate_decoder)(x)
            x = tf.keras.layers.Activation(self._decoder_activation_function)(x)

        # Output layer
        x = Dense(output_shape, kernel_initializer=initialization, name="Output_1")(x)
        decoder_output = tf.keras.layers.Activation(self._decoder_last_layer_activation)(x)

        return Model([neural_model_inputs, label_input], decoder_output, name="Decoder")

    @property
    def dropout_decay_rate_decoder(self) -> float:
        """Gets the dropout decay rate."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self) -> List[int]:
        """Gets the number of neurons in decoder layers."""
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_decoder: float) -> None:
        """Sets the dropout decay rate."""
        if not isinstance(dropout_decay_rate_decoder, (int, float)) or not (0 <= dropout_decay_rate_decoder <= 1):
            raise ValueError(f"Invalid dropout rate: {dropout_decay_rate_decoder}. Must be between 0 and 1.")
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_decoder


class VanillaDecoderPyTorch:
    """PyTorch implementation of VanillaDecoder."""

    def __init__(self, 
                 latent_dimension: int, 
                 output_shape: int, 
                 activation_function: str, 
                 initializer_mean: float,
                 initializer_deviation: float, 
                 dropout_decay_decoder: float, 
                 last_layer_activation: str,
                 number_neurons_decoder: List[int], 
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[Dict] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function, initializer_mean,
                                 initializer_deviation, dropout_decay_decoder, last_layer_activation,
                                 number_neurons_decoder, dataset_type, number_samples_per_class)

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_activation_function = activation_function.lower()
        self._decoder_last_layer_activation = last_layer_activation.lower()
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class

    def _validate_parameters(self, latent_dimension, output_shape, activation_function, initializer_mean,
                           initializer_deviation, dropout_decay_decoder, last_layer_activation,
                           number_neurons_decoder, dataset_type, number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid latent_dimension: {latent_dimension}. Must be a positive integer.")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid output_shape: {output_shape}. Must be a positive integer.")
        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid activation_function: {activation_function}. Must be a string.")
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid initializer_mean: {initializer_mean}. Must be a number.")
        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError(f"Invalid initializer_deviation: {initializer_deviation}. Must be a number.")
        if not isinstance(dropout_decay_decoder, (int, float)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError(f"Invalid dropout_decay_decoder: {dropout_decay_decoder}. Must be between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid last_layer_activation: {last_layer_activation}. Must be a string.")
        if not isinstance(number_neurons_decoder, list) or not all(isinstance(x, int) and x > 0 for x in number_neurons_decoder):
            raise ValueError(f"Invalid number_neurons_decoder: {number_neurons_decoder}. Must be a list of positive integers.")
        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be a valid type.")
        if number_samples_per_class is not None and (not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(f"Invalid number_samples_per_class: {number_samples_per_class}. Must be a dict with 'number_classes'.")

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
            nn.init.normal_(module.weight, mean=self._decoder_initializer_mean,
                          std=self._decoder_initializer_deviation)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_decoder(self, output_shape: int):
        """
        Constructs and returns the decoder model (PyTorch).

        Args:
            output_shape (int): The output dimensionality of the decoder.

        Returns:
            nn.Module: The constructed decoder model.
        """
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid output_shape: {output_shape}. Must be a positive integer.")

        class ConditionalDecoder(nn.Module):
            def __init__(self, latent_dim, n_classes, layer_sizes, output_dim,
                        dropout_rate, activation_fn, last_activation_fn, init_fn):
                super(ConditionalDecoder, self).__init__()
                
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = nn.ModuleList()
                
                in_features = latent_dim + n_classes
                
                # Build hidden layers
                for layer_size in layer_sizes:
                    self.layers.append(nn.Linear(in_features, layer_size))
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    self.activations.append(activation_fn)
                    in_features = layer_size
                
                # Output layer
                self.output_layer = nn.Linear(in_features, output_dim)
                self.last_activation = last_activation_fn
                
                self.apply(init_fn)
            
            def forward(self, latent_input, label_input):
                x = torch.cat([latent_input, label_input], dim=1)
                
                for layer, dropout, activation in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = activation(x)
                
                x = self.output_layer(x)
                decoded = self.last_activation(x)
                
                return decoded

        decoder = ConditionalDecoder(
            self._decoder_latent_dimension,
            self._decoder_number_samples_per_class["number_classes"],
            self._decoder_number_neurons_decoder,
            output_shape,
            self._decoder_dropout_decay_rate_decoder,
            self._get_activation(self._decoder_activation_function),
            self._get_activation(self._decoder_last_layer_activation),
            self._initialize_weights
        )

        return decoder

    @property
    def dropout_decay_rate_decoder(self) -> float:
        """Gets the dropout decay rate."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self) -> List[int]:
        """Gets the number of neurons in decoder layers."""
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_decoder: float) -> None:
        """Sets the dropout decay rate."""
        if not isinstance(dropout_decay_rate_decoder, (int, float)) or not (0 <= dropout_decay_rate_decoder <= 1):
            raise ValueError(f"Invalid dropout rate: {dropout_decay_rate_decoder}. Must be between 0 and 1.")
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_decoder


def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
