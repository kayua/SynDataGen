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

# Detect available framework
FRAMEWORK = None
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate
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


class VanillaGenerator:
    """
    VanillaGenerator (Framework Agnostic)

    Implements a dense generator model for generating synthetic data using a
    vanilla architecture. This class is designed for generating synthetic data
    from a latent space using a fully connected neural network. It supports
    flexible configurations for the generator layers, activations, and dropout
    rates, with the option for conditional generation based on the number of
    samples per class.

    This implementation automatically detects and uses either TensorFlow or PyTorch.

    Attributes:
        latent_dimension (int):
            The dimensionality of the latent space, which serves as the input to the generator.
        output_shape (int):
            The desired dimension of the generated output data.
        activation_function (str):
            The activation function used in intermediate layers (e.g., 'relu', 'leaky_relu').
        initializer_mean (float):
            The mean for the weight initialization.
        initializer_deviation (float):
            The standard deviation for the weight initialization.
        dropout_decay_rate_g (float):
            The rate at which the dropout is applied in generator layers, should be between 0.0 and 1.0.
        last_layer_activation (str):
            The activation function to be applied in the last layer (e.g., 'sigmoid' or 'tanh').
        dense_layer_sizes_g (list):
            A list of integers representing the number of units in each dense layer of the generator.
        dataset_type (type):
            The data type for the input tensors (default is numpy.float32).
        number_samples_per_class (dict | None):
            An optional dictionary indicating the number of samples per class for conditional data generation.

    Example:
        >>> generator = VanillaGenerator(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function="relu",
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation="sigmoid",
        ...     dense_layer_sizes_g=[128, 256, 512],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = generator.get_generator()
    """

    def __new__(cls, *args, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Returns:
            VanillaGeneratorTF or VanillaGeneratorPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return object.__new__(VanillaGeneratorTF)
        elif FRAMEWORK == 'pytorch':
            return object.__new__(VanillaGeneratorPyTorch)


class VanillaGeneratorTF:
    """TensorFlow implementation of VanillaGenerator."""

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: list,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict | None = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, last_layer_activation,
                                 number_samples_per_class)

        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function.lower()
        self._generator_last_layer_activation = last_layer_activation.lower()
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, last_layer_activation,
                           number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")
        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")
        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")
        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

    def _get_activation(self, activation_name: str):
        """Returns the appropriate TensorFlow activation function."""
        activation_map = {
            'relu': tf.nn.relu,
            'leaky_relu': tf.nn.leaky_relu,
            'sigmoid': tf.nn.sigmoid,
            'tanh': tf.nn.tanh,
            'elu': tf.nn.elu,
            'selu': tf.nn.selu
        }
        return activation_map.get(activation_name, tf.nn.relu)

    def get_generator(self) -> Model:
        """
        Builds and returns the generator model (TensorFlow).

        Returns:
            Model: A Keras model with inputs for latent vectors and conditional labels.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("`number_samples_per_class` must include a 'number_classes' key.")

        initialization = RandomNormal(mean=self._generator_initializer_mean, 
                                     stddev=self._generator_initializer_deviation)
        
        neural_model_inputs = Input(shape=(self._generator_latent_dimension,), 
                                    dtype=self._generator_dataset_type)
        latent_input = Input(shape=(self._generator_latent_dimension,))
        label_input = Input(shape=(self._generator_number_samples_per_class["number_classes"],), 
                          dtype=self._generator_dataset_type)

        # Build dense generator model
        x = Dense(self._generator_dense_layer_sizes_g[0], kernel_initializer=initialization)(neural_model_inputs)
        x = Dropout(self._generator_dropout_decay_rate_g)(x)
        x = tf.keras.layers.Activation(self._generator_activation_function)(x)

        for layer_size in self._generator_dense_layer_sizes_g[1:]:
            x = Dense(layer_size, kernel_initializer=initialization)(x)
            x = Dropout(self._generator_dropout_decay_rate_g)(x)
            x = tf.keras.layers.Activation(self._generator_activation_function)(x)

        x = Dense(self._generator_output_shape, kernel_initializer=initialization)(x)
        generator_output = tf.keras.layers.Activation(self._generator_last_layer_activation)(x)
        
        generator_model = Model(neural_model_inputs, generator_output, name="Dense_Generator")
        self._generator_model_dense = generator_model

        # Concatenate latent input with label input for conditional generation
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._generator_latent_dimension)(label_embedding)
        model_input = tf.keras.layers.Activation(self._generator_activation_function)(model_input)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def get_dense_generator_model(self) -> Model | None:
        """Returns the standalone dense generator model."""
        return self._generator_model_dense

    def set_dropout_decay_rate_generator(self, dropout_decay_rate_generator: float) -> None:
        """Updates the dropout rate of the generator."""
        if not (0.0 <= dropout_decay_rate_generator <= 1.0):
            raise ValueError("`dropout_decay_rate_generator` must be between 0.0 and 1.0.")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    def set_dense_layer_sizes_generator(self, dense_layer_sizes_generator: list) -> None:
        """Updates the dense layer sizes of the generator."""
        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("`dense_layer_sizes_generator` must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator


class VanillaGeneratorPyTorch:
    """PyTorch implementation of VanillaGenerator."""

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: list,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict | None = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, last_layer_activation,
                                 number_samples_per_class)

        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function.lower()
        self._generator_last_layer_activation = last_layer_activation.lower()
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, last_layer_activation,
                           number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")
        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")
        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")
        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

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
            nn.init.normal_(module.weight, mean=self._generator_initializer_mean, 
                          std=self._generator_initializer_deviation)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_generator(self):
        """
        Builds and returns the generator model (PyTorch).

        Returns:
            nn.Module: A PyTorch model for conditional generation.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("`number_samples_per_class` must include a 'number_classes' key.")

        class DenseGenerator(nn.Module):
            def __init__(self, latent_dim, output_dim, layer_sizes, dropout_rate, 
                        activation_fn, last_activation_fn):
                super(DenseGenerator, self).__init__()
                layers = []
                
                in_features = latent_dim
                for layer_size in layer_sizes:
                    layers.append(nn.Linear(in_features, layer_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(activation_fn)
                    in_features = layer_size
                
                layers.append(nn.Linear(in_features, output_dim))
                layers.append(last_activation_fn)
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)

        class ConditionalGenerator(nn.Module):
            def __init__(self, latent_dim, n_classes, dense_gen, activation_fn):
                super(ConditionalGenerator, self).__init__()
                self.dense_gen = dense_gen
                self.label_embedding = nn.Linear(latent_dim + n_classes, latent_dim)
                self.activation = activation_fn
            
            def forward(self, latent, labels):
                x = torch.cat([latent, labels], dim=1)
                x = self.label_embedding(x)
                x = self.activation(x)
                return self.dense_gen(x)

        dense_gen = DenseGenerator(
            self._generator_latent_dimension,
            self._generator_output_shape,
            self._generator_dense_layer_sizes_g,
            self._generator_dropout_decay_rate_g,
            self._get_activation(self._generator_activation_function),
            self._get_activation(self._generator_last_layer_activation)
        )
        dense_gen.apply(self._initialize_weights)
        self._generator_model_dense = dense_gen

        conditional_gen = ConditionalGenerator(
            self._generator_latent_dimension,
            self._generator_number_samples_per_class["number_classes"],
            dense_gen,
            self._get_activation(self._generator_activation_function)
        )
        conditional_gen.apply(self._initialize_weights)

        return conditional_gen

    def get_dense_generator_model(self):
        """Returns the standalone dense generator model."""
        return self._generator_model_dense

    def set_dropout_decay_rate_generator(self, dropout_decay_rate_generator: float) -> None:
        """Updates the dropout rate of the generator."""
        if not (0.0 <= dropout_decay_rate_generator <= 1.0):
            raise ValueError("`dropout_decay_rate_generator` must be between 0.0 and 1.0.")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    def set_dense_layer_sizes_generator(self, dense_layer_sizes_generator: list) -> None:
        """Updates the dense layer sizes of the generator."""
        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("`dense_layer_sizes_generator` must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator


def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
