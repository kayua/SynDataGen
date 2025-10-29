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


class VanillaDiscriminator:
    """
    VanillaDiscriminator (Framework Agnostic)

    Implements a fully-connected (dense) discriminator network for use in generative models,
    such as Generative Adversarial Networks (GANs). This class provides flexibility in the design
    of the architecture, including customizable latent dimensions, output shapes, activation functions,
    dropout rates, and layer sizes.

    This implementation automatically detects and uses either TensorFlow or PyTorch.

    Attributes:
        latent_dimension (int):
            Dimensionality of the input latent space for the discriminator network.
        output_shape (int):
            The output shape of the network, typically used to define the shape of input data.
        activation_function (str):
            The activation function applied to all hidden layers (e.g., 'relu', 'leaky_relu').
        last_layer_activation (str):
            The activation function applied to the last layer (e.g., 'sigmoid').
        dropout_decay_rate_d (float):
            Dropout rate applied to layers in the network to help prevent overfitting.
        dense_layer_sizes_d (List[int]):
            List of integers defining the number of units in each dense layer.
        dataset_type (numpy.dtype):
            The data type of the dataset (default: numpy.float32).
        initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing the number of samples per class.

    Example:
        >>> discriminator = VanillaDiscriminator(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function='leaky_relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_d=0.3,
        ...     last_layer_activation='sigmoid',
        ...     dense_layer_sizes_d=[512, 256, 128],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = discriminator.get_discriminator()
    """

    def __new__(cls, *args, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Returns:
            VanillaDiscriminatorTF or VanillaDiscriminatorPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return object.__new__(VanillaDiscriminatorTF)
        elif FRAMEWORK == 'pytorch':
            return object.__new__(VanillaDiscriminatorPyTorch)


class VanillaDiscriminatorTF:
    """TensorFlow implementation of VanillaDiscriminator."""

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_d: float,
                 last_layer_activation: str,
                 dense_layer_sizes_d: List[int],
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, dropout_decay_rate_d,
                                 last_layer_activation, dense_layer_sizes_d, number_samples_per_class)

        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function.lower()
        self._discriminator_last_layer_activation = last_layer_activation.lower()
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, dropout_decay_rate_d,
                           last_layer_activation, dense_layer_sizes_d, number_samples_per_class):
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
        if not isinstance(dropout_decay_rate_d, (float, int)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError("dropout_decay_rate_d must be a float between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")
        if not isinstance(dense_layer_sizes_d, list) or not all(isinstance(n, int) and n > 0 for n in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a list of positive integers.")
        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

    def get_discriminator(self) -> Model:
        """
        Build and return the complete discriminator model (TensorFlow).

        Returns:
            Model: A Keras model representing the discriminator.
        """
        if not self._discriminator_number_samples_per_class or "number_classes" not in self._discriminator_number_samples_per_class:
            raise ValueError("`number_samples_per_class` must include a 'number_classes' key.")

        initialization = RandomNormal(mean=self._discriminator_initializer_mean,
                                     stddev=self._discriminator_initializer_deviation)

        # Define the input layers
        neural_model_input = Input(shape=(self._discriminator_output_shape,), 
                                  dtype=self._discriminator_dataset_type)
        discriminator_shape_input = Input(shape=(self._discriminator_output_shape,))
        label_input = Input(shape=(self._discriminator_number_samples_per_class["number_classes"],), 
                          dtype=self._discriminator_dataset_type)

        # Build the discriminator model
        x = Dense(self._discriminator_dense_layer_sizes_d[0], kernel_initializer=initialization)(neural_model_input)
        x = Dropout(self._discriminator_dropout_decay_rate_d)(x)
        x = tf.keras.layers.Activation(self._discriminator_activation_function)(x)

        # Add additional dense layers with dropout and activations
        for layer_size in self._discriminator_dense_layer_sizes_d[1:]:
            x = Dense(layer_size, kernel_initializer=initialization)(x)
            x = Dropout(self._discriminator_dropout_decay_rate_d)(x)
            x = tf.keras.layers.Activation(self._discriminator_activation_function)(x)

        # Final output layer
        x = Dense(1, kernel_initializer=initialization)(x)
        discriminator_output = tf.keras.layers.Activation(self._discriminator_last_layer_activation)(x)
        
        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_output, name="Dense_Discriminator")
        self._discriminator_model_dense = discriminator_model

        # Concatenate the input label and shape input
        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._discriminator_output_shape, kernel_initializer=initialization)(label_embedding)
        model_input = tf.keras.layers.Activation(self._discriminator_activation_function)(model_input)

        # Get the final output of the discriminator model
        validity = discriminator_model(model_input)

        return Model(inputs=[discriminator_shape_input, label_input], outputs=validity, name='Discriminator')

    def get_dense_discriminator_model(self) -> Optional[Model]:
        """Returns the standalone dense discriminator model."""
        return self._discriminator_model_dense

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float) -> None:
        """Updates the dropout decay rate for the discriminator network."""
        if not (0.0 <= dropout_decay_rate_discriminator <= 1.0):
            raise ValueError("`dropout_decay_rate_discriminator` must be between 0.0 and 1.0.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]) -> None:
        """Updates the sizes for the dense layers in the discriminator network."""
        if not dense_layer_sizes_discriminator or any(size <= 0 for size in dense_layer_sizes_discriminator):
            raise ValueError("`dense_layer_sizes_discriminator` must be a list of positive integers.")
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator


class VanillaDiscriminatorPyTorch:
    """PyTorch implementation of VanillaDiscriminator."""

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_d: float,
                 last_layer_activation: str,
                 dense_layer_sizes_d: List[int],
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, dropout_decay_rate_d,
                                 last_layer_activation, dense_layer_sizes_d, number_samples_per_class)

        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function.lower()
        self._discriminator_last_layer_activation = last_layer_activation.lower()
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, dropout_decay_rate_d,
                           last_layer_activation, dense_layer_sizes_d, number_samples_per_class):
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
        if not isinstance(dropout_decay_rate_d, (float, int)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError("dropout_decay_rate_d must be a float between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")
        if not isinstance(dense_layer_sizes_d, list) or not all(isinstance(n, int) and n > 0 for n in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a list of positive integers.")
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
            nn.init.normal_(module.weight, mean=self._discriminator_initializer_mean,
                          std=self._discriminator_initializer_deviation)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_discriminator(self):
        """
        Build and return the complete discriminator model (PyTorch).

        Returns:
            nn.Module: A PyTorch model representing the discriminator.
        """
        if not self._discriminator_number_samples_per_class or "number_classes" not in self._discriminator_number_samples_per_class:
            raise ValueError("`number_samples_per_class` must include a 'number_classes' key.")

        class DenseDiscriminator(nn.Module):
            def __init__(self, input_dim, layer_sizes, dropout_rate, activation_fn, last_activation_fn):
                super(DenseDiscriminator, self).__init__()
                layers = []
                
                in_features = input_dim
                for layer_size in layer_sizes:
                    layers.append(nn.Linear(in_features, layer_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(activation_fn)
                    in_features = layer_size
                
                layers.append(nn.Linear(in_features, 1))
                layers.append(last_activation_fn)
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)

        class ConditionalDiscriminator(nn.Module):
            def __init__(self, output_shape, n_classes, dense_disc, activation_fn):
                super(ConditionalDiscriminator, self).__init__()
                self.dense_disc = dense_disc
                self.label_embedding = nn.Linear(output_shape + n_classes, output_shape)
                self.activation = activation_fn
            
            def forward(self, shape_input, labels):
                x = torch.cat([shape_input, labels], dim=1)
                x = self.label_embedding(x)
                x = self.activation(x)
                return self.dense_disc(x)

        dense_disc = DenseDiscriminator(
            self._discriminator_output_shape,
            self._discriminator_dense_layer_sizes_d,
            self._discriminator_dropout_decay_rate_d,
            self._get_activation(self._discriminator_activation_function),
            self._get_activation(self._discriminator_last_layer_activation)
        )
        dense_disc.apply(self._initialize_weights)
        self._discriminator_model_dense = dense_disc

        conditional_disc = ConditionalDiscriminator(
            self._discriminator_output_shape,
            self._discriminator_number_samples_per_class["number_classes"],
            dense_disc,
            self._get_activation(self._discriminator_activation_function)
        )
        conditional_disc.apply(self._initialize_weights)

        return conditional_disc

    def get_dense_discriminator_model(self):
        """Returns the standalone dense discriminator model."""
        return self._discriminator_model_dense

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float) -> None:
        """Updates the dropout decay rate for the discriminator network."""
        if not (0.0 <= dropout_decay_rate_discriminator <= 1.0):
            raise ValueError("`dropout_decay_rate_discriminator` must be between 0.0 and 1.0.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]) -> None:
        """Updates the sizes for the dense layers in the discriminator network."""
        if not dense_layer_sizes_discriminator or any(size <= 0 for size in dense_layer_sizes_discriminator):
            raise ValueError("`dense_layer_sizes_discriminator` must be a list of positive integers.")
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator


def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
