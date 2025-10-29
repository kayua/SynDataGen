#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/10/29'
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

import sys
import numpy
from typing import List, Dict, Optional, Tuple

# Detect available framework
FRAMEWORK = None
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Flatten
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
    such as GANs or WGANs. This class supports fully customizable layer sizes, activation
    functions, dropout rates, and initialization schemes, allowing it to be adapted to
    various tasks requiring a critic or discriminator network.

    This implementation automatically detects and uses either TensorFlow or PyTorch.

    Attributes:
        latent_dimension (int):
            Dimensionality of the latent space used by the model.
        output_shape (Tuple[int, ...] or int):
            Shape of the expected output data (e.g., for image discrimination).
        activation_function (str):
            Activation function applied to all hidden layers.
        last_layer_activation (str):
            Activation function applied to the final output layer.
        dropout_decay_rate_d (float):
            Dropout rate applied to dense layers to improve generalization.
        dense_layer_sizes_d (List[int]):
            List of integers specifying the number of units in each dense layer.
        dataset_type (type):
            Data type of the input dataset (default: numpy.float32).
        initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing metadata about class distribution.
            Must include a key "number_classes" if provided.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).
          Generative Adversarial Networks. arXiv preprint arXiv:1406.2661.
          Available at: https://arxiv.org/abs/1406.2661

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
        ...     dataset_type=numpy.float32,
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
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[Dict] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, dropout_decay_rate_d,
                                 last_layer_activation, dense_layer_sizes_d, dataset_type,
                                 number_samples_per_class)

        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function.lower()
        self._discriminator_last_layer_activation = last_layer_activation.lower()
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, dropout_decay_rate_d,
                           last_layer_activation, dense_layer_sizes_d, dataset_type,
                           number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid latent_dimension: {latent_dimension}. Must be a positive integer.")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid output_shape: {output_shape}. Must be a positive integer.")
        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid activation_function: {activation_function}. Must be a string.")
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid initializer_mean: {initializer_mean}. Must be a number.")
        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError(f"Invalid initializer_deviation: {initializer_deviation}. Must be positive.")
        if not isinstance(dropout_decay_rate_d, (int, float)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError(f"Invalid dropout_decay_rate_d: {dropout_decay_rate_d}. Must be between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid last_layer_activation: {last_layer_activation}. Must be a string.")
        if not isinstance(dense_layer_sizes_d, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_d):
            raise ValueError(f"Invalid dense_layer_sizes_d: {dense_layer_sizes_d}. Must be a list of positive integers.")
        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be a valid type.")
        if number_samples_per_class is not None and (not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(f"Invalid number_samples_per_class: {number_samples_per_class}. Must be a dict with 'number_classes'.")

    def get_discriminator(self) -> Model:
        """
        Constructs the discriminator model using dense layers (TensorFlow).

        Returns:
            keras.Model: A Keras Model instance representing the discriminator.

        Raises:
            ValueError: If number_samples_per_class is not properly defined.
        """
        if not self._discriminator_number_samples_per_class or "number_classes" not in self._discriminator_number_samples_per_class:
            raise ValueError("number_samples_per_class with a 'number_classes' key must be provided.")

        initialization = RandomNormal(mean=self._discriminator_initializer_mean,
                                     stddev=self._discriminator_initializer_deviation)

        # Define inputs
        neural_model_input = Input(shape=(self._discriminator_output_shape,),
                                  dtype=self._discriminator_dataset_type)
        discriminator_shape_input = Input(shape=(self._discriminator_output_shape,))
        label_input = Input(shape=(self._discriminator_number_samples_per_class["number_classes"],),
                          dtype=self._discriminator_dataset_type)

        # Build dense discriminator
        x = Dense(self._discriminator_dense_layer_sizes_d[0],
                 kernel_initializer=initialization)(neural_model_input)
        x = Dropout(self._discriminator_dropout_decay_rate_d)(x)
        x = tf.keras.layers.Activation(self._discriminator_activation_function)(x)

        for layer_size in self._discriminator_dense_layer_sizes_d[1:]:
            x = Dense(layer_size, kernel_initializer=initialization)(x)
            x = Dropout(self._discriminator_dropout_decay_rate_d)(x)
            x = tf.keras.layers.Activation(self._discriminator_activation_function)(x)

        x = Dense(1)(x)
        discriminator_output = tf.keras.layers.Activation(self._discriminator_last_layer_activation)(x)

        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_output,
                                   name="Dense_Discriminator")
        self._discriminator_model_dense = discriminator_model

        # Concatenate label information
        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._discriminator_output_shape,
                          kernel_initializer=initialization)(label_embedding)

        validity = discriminator_model(model_input)

        return Model(inputs=[discriminator_shape_input, label_input], outputs=validity,
                    name="Discriminator")

    @property
    def dense_discriminator_model(self) -> Optional[Model]:
        """Returns the dense part of the discriminator model."""
        return self._discriminator_model_dense

    @property
    def dropout_decay_rate_discriminator(self) -> float:
        """Gets the dropout rate for the discriminator."""
        return self._discriminator_dropout_decay_rate_d

    @property
    def dense_layer_sizes_discriminator(self) -> List[int]:
        """Gets the sizes of the dense layers for the discriminator."""
        return self._discriminator_dense_layer_sizes_d

    @dropout_decay_rate_discriminator.setter
    def dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float) -> None:
        """Sets the dropout rate for the discriminator."""
        if not isinstance(dropout_decay_rate_discriminator, (int, float)) or not (0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError(f"Invalid dropout rate: {dropout_decay_rate_discriminator}. Must be between 0 and 1.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    @dense_layer_sizes_discriminator.setter
    def dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]) -> None:
        """Sets the sizes of the dense layers for the discriminator."""
        if not isinstance(dense_layer_sizes_discriminator, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_discriminator):
            raise ValueError(f"Invalid dense layer sizes: {dense_layer_sizes_discriminator}. Must be a list of positive integers.")
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
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[Dict] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, dropout_decay_rate_d,
                                 last_layer_activation, dense_layer_sizes_d, dataset_type,
                                 number_samples_per_class)

        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function.lower()
        self._discriminator_last_layer_activation = last_layer_activation.lower()
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, dropout_decay_rate_d,
                           last_layer_activation, dense_layer_sizes_d, dataset_type,
                           number_samples_per_class):
        """Validates initialization parameters."""
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid latent_dimension: {latent_dimension}. Must be a positive integer.")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid output_shape: {output_shape}. Must be a positive integer.")
        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid activation_function: {activation_function}. Must be a string.")
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid initializer_mean: {initializer_mean}. Must be a number.")
        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError(f"Invalid initializer_deviation: {initializer_deviation}. Must be positive.")
        if not isinstance(dropout_decay_rate_d, (int, float)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError(f"Invalid dropout_decay_rate_d: {dropout_decay_rate_d}. Must be between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid last_layer_activation: {last_layer_activation}. Must be a string.")
        if not isinstance(dense_layer_sizes_d, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_d):
            raise ValueError(f"Invalid dense_layer_sizes_d: {dense_layer_sizes_d}. Must be a list of positive integers.")
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
            nn.init.normal_(module.weight, mean=self._discriminator_initializer_mean,
                          std=self._discriminator_initializer_deviation)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_discriminator(self):
        """
        Constructs the discriminator model using dense layers (PyTorch).

        Returns:
            nn.Module: The constructed discriminator model.

        Raises:
            ValueError: If number_samples_per_class is not properly defined.
        """
        if not self._discriminator_number_samples_per_class or "number_classes" not in self._discriminator_number_samples_per_class:
            raise ValueError("number_samples_per_class with a 'number_classes' key must be provided.")

        class DenseDiscriminator(nn.Module):
            def __init__(self, input_dim, layer_sizes, dropout_rate, activation_fn,
                        last_activation_fn, init_fn):
                super(DenseDiscriminator, self).__init__()
                
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = nn.ModuleList()
                
                in_features = input_dim
                
                # Build hidden layers
                for layer_size in layer_sizes:
                    self.layers.append(nn.Linear(in_features, layer_size))
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    self.activations.append(activation_fn)
                    in_features = layer_size
                
                # Output layer (1 neuron for validity)
                self.output_layer = nn.Linear(in_features, 1)
                self.last_activation = last_activation_fn
                
                self.apply(init_fn)
            
            def forward(self, x):
                for layer, dropout, activation in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = activation(x)
                
                x = self.output_layer(x)
                validity = self.last_activation(x)
                
                return validity

        class ConditionalDiscriminator(nn.Module):
            def __init__(self, output_dim, n_classes, dense_discriminator, init_mean, init_std):
                super(ConditionalDiscriminator, self).__init__()
                
                self.embedding_layer = nn.Linear(output_dim + n_classes, output_dim)
                self.dense_discriminator = dense_discriminator
                
                # Initialize embedding layer
                nn.init.normal_(self.embedding_layer.weight, mean=init_mean, std=init_std)
                if self.embedding_layer.bias is not None:
                    nn.init.constant_(self.embedding_layer.bias, 0)
            
            def forward(self, shape_input, label_input):
                x = torch.cat([shape_input, label_input], dim=1)
                x = self.embedding_layer(x)
                validity = self.dense_discriminator(x)
                
                return validity

        # Create dense discriminator
        dense_discriminator = DenseDiscriminator(
            self._discriminator_output_shape,
            self._discriminator_dense_layer_sizes_d,
            self._discriminator_dropout_decay_rate_d,
            self._get_activation(self._discriminator_activation_function),
            self._get_activation(self._discriminator_last_layer_activation),
            self._initialize_weights
        )

        self._discriminator_model_dense = dense_discriminator

        # Create conditional discriminator
        discriminator = ConditionalDiscriminator(
            self._discriminator_output_shape,
            self._discriminator_number_samples_per_class["number_classes"],
            dense_discriminator,
            self._discriminator_initializer_mean,
            self._discriminator_initializer_deviation
        )

        return discriminator

    @property
    def dense_discriminator_model(self):
        """Returns the dense part of the discriminator model."""
        return self._discriminator_model_dense

    @property
    def dropout_decay_rate_discriminator(self) -> float:
        """Gets the dropout rate for the discriminator."""
        return self._discriminator_dropout_decay_rate_d

    @property
    def dense_layer_sizes_discriminator(self) -> List[int]:
        """Gets the sizes of the dense layers for the discriminator."""
        return self._discriminator_dense_layer_sizes_d

    @dropout_decay_rate_discriminator.setter
    def dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float) -> None:
        """Sets the dropout rate for the discriminator."""
        if not isinstance(dropout_decay_rate_discriminator, (int, float)) or not (0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError(f"Invalid dropout rate: {dropout_decay_rate_discriminator}. Must be between 0 and 1.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    @dense_layer_sizes_discriminator.setter
    def dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]) -> None:
        """Sets the sizes of the dense layers for the discriminator."""
        if not isinstance(dense_layer_sizes_discriminator, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_discriminator):
            raise ValueError(f"Invalid dense layer sizes: {dense_layer_sizes_discriminator}. Must be a list of positive integers.")
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator


def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
