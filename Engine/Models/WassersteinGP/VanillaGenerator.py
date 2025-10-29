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
from typing import List, Dict, Optional

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


class VanillaGenerator:
    """
    VanillaGenerator (Framework Agnostic)

    Implements a fully connected (dense) generator model for use in generative models,
    such as GANs. This generator is designed to work with label conditioning and
    supports customization of activation functions, layer sizes, initialization, and
    other hyperparameters.

    This implementation automatically detects and uses either TensorFlow or PyTorch.

    Attributes:
        latent_dimension (int):
            Dimensionality of the input latent space.
        output_shape (int):
            Dimensionality of the generated output data.
        activation_function (str):
            Activation function applied to all hidden layers.
        last_layer_activation (str):
            Activation function applied to the final output layer.
        dropout_decay_rate_g (float):
            Dropout rate applied to dense layers to improve generalization.
        dense_layer_sizes_g (List[int]):
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
        >>> generator = VanillaGenerator(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function='leaky_relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation='tanh',
        ...     dense_layer_sizes_g=[256, 512, 1024],
        ...     dataset_type=numpy.float32,
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

    def __init__(self, latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: List[int],
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[Dict] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, dropout_decay_rate_g,
                                 last_layer_activation, dense_layer_sizes_g, dataset_type,
                                 number_samples_per_class)

        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function.lower()
        self._generator_last_layer_activation = last_layer_activation.lower()
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, dropout_decay_rate_g,
                           last_layer_activation, dense_layer_sizes_g, dataset_type,
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
        if not isinstance(dropout_decay_rate_g, (int, float)) or not (0 <= dropout_decay_rate_g <= 1):
            raise ValueError(f"Invalid dropout_decay_rate_g: {dropout_decay_rate_g}. Must be between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid last_layer_activation: {last_layer_activation}. Must be a string.")
        if not isinstance(dense_layer_sizes_g, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_g):
            raise ValueError(f"Invalid dense_layer_sizes_g: {dense_layer_sizes_g}. Must be a list of positive integers.")
        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be a valid type.")
        if number_samples_per_class is not None and (not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(f"Invalid number_samples_per_class: {number_samples_per_class}. Must be a dict with 'number_classes'.")

    def get_generator(self) -> Model:
        """
        Constructs and returns the generator model (TensorFlow).

        Returns:
            keras.Model: A Keras model implementing the generator with latent and label inputs.

        Raises:
            ValueError: If number_samples_per_class is not properly defined.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("Number of samples per class must include 'number_classes'.")

        initialization = RandomNormal(mean=self._generator_initializer_mean,
                                     stddev=self._generator_initializer_deviation)

        # Define inputs
        neural_model_inputs = Input(shape=(self._generator_latent_dimension,),
                                   dtype=self._generator_dataset_type)
        latent_input = Input(shape=(self._generator_latent_dimension,))
        label_input = Input(shape=(self._generator_number_samples_per_class["number_classes"],),
                          dtype=self._generator_dataset_type)

        # Dense generator model
        x = Dense(self._generator_dense_layer_sizes_g[0],
                 kernel_initializer=initialization)(neural_model_inputs)
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

        # Concatenate label information
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._generator_latent_dimension)(label_embedding)
        model_input = tf.keras.layers.Activation(self._generator_activation_function)(model_input)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    @property
    def dense_generator_model(self) -> Optional[Model]:
        """Property that retrieves the dense generator submodel without label conditioning."""
        return self._generator_model_dense

    @property
    def dropout_decay_rate_generator(self) -> float:
        """Property to get the dropout decay rate for the generator."""
        return self._generator_dropout_decay_rate_g

    @property
    def dense_layer_sizes_generator(self) -> List[int]:
        """Property to get the dense layer sizes for the generator."""
        return self._generator_dense_layer_sizes_g

    @dropout_decay_rate_generator.setter
    def dropout_decay_rate_generator(self, dropout_decay_rate_generator: float):
        """Property to set the dropout decay rate for the generator."""
        if not isinstance(dropout_decay_rate_generator, (int, float)) or not (0 <= dropout_decay_rate_generator <= 1):
            raise ValueError(f"Invalid dropout rate: {dropout_decay_rate_generator}. Must be between 0 and 1.")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    @dense_layer_sizes_generator.setter
    def dense_layer_sizes_generator(self, dense_layer_sizes_generator: List[int]):
        """Property to set the dense layer sizes for the generator."""
        if not isinstance(dense_layer_sizes_generator, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_generator):
            raise ValueError(f"Invalid dense layer sizes: {dense_layer_sizes_generator}. Must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator


class VanillaGeneratorPyTorch:
    """PyTorch implementation of VanillaGenerator."""

    def __init__(self, latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: List[int],
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[Dict] = None):
        
        self._validate_parameters(latent_dimension, output_shape, activation_function,
                                 initializer_mean, initializer_deviation, dropout_decay_rate_g,
                                 last_layer_activation, dense_layer_sizes_g, dataset_type,
                                 number_samples_per_class)

        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function.lower()
        self._generator_last_layer_activation = last_layer_activation.lower()
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_model_dense = None

    def _validate_parameters(self, latent_dimension, output_shape, activation_function,
                           initializer_mean, initializer_deviation, dropout_decay_rate_g,
                           last_layer_activation, dense_layer_sizes_g, dataset_type,
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
        if not isinstance(dropout_decay_rate_g, (int, float)) or not (0 <= dropout_decay_rate_g <= 1):
            raise ValueError(f"Invalid dropout_decay_rate_g: {dropout_decay_rate_g}. Must be between 0 and 1.")
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid last_layer_activation: {last_layer_activation}. Must be a string.")
        if not isinstance(dense_layer_sizes_g, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_g):
            raise ValueError(f"Invalid dense_layer_sizes_g: {dense_layer_sizes_g}. Must be a list of positive integers.")
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
            nn.init.normal_(module.weight, mean=self._generator_initializer_mean,
                          std=self._generator_initializer_deviation)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_generator(self):
        """
        Constructs and returns the generator model (PyTorch).

        Returns:
            nn.Module: The constructed generator model.

        Raises:
            ValueError: If number_samples_per_class is not properly defined.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("Number of samples per class must include 'number_classes'.")

        class DenseGenerator(nn.Module):
            def __init__(self, latent_dim, layer_sizes, output_dim,
                        dropout_rate, activation_fn, last_activation_fn, init_fn):
                super(DenseGenerator, self).__init__()
                
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = nn.ModuleList()
                
                in_features = latent_dim
                
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
            
            def forward(self, x):
                for layer, dropout, activation in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = activation(x)
                
                x = self.output_layer(x)
                output = self.last_activation(x)
                
                return output

        class ConditionalGenerator(nn.Module):
            def __init__(self, latent_dim, n_classes, dense_generator, activation_fn):
                super(ConditionalGenerator, self).__init__()
                
                self.embedding_layer = nn.Linear(latent_dim + n_classes, latent_dim)
                self.embedding_activation = activation_fn
                self.dense_generator = dense_generator
            
            def forward(self, latent_input, label_input):
                x = torch.cat([latent_input, label_input], dim=1)
                x = self.embedding_layer(x)
                x = self.embedding_activation(x)
                output = self.dense_generator(x)
                
                return output

        # Create dense generator
        dense_generator = DenseGenerator(
            self._generator_latent_dimension,
            self._generator_dense_layer_sizes_g,
            self._generator_output_shape,
            self._generator_dropout_decay_rate_g,
            self._get_activation(self._generator_activation_function),
            self._get_activation(self._generator_last_layer_activation),
            self._initialize_weights
        )

        self._generator_model_dense = dense_generator

        # Create conditional generator
        generator = ConditionalGenerator(
            self._generator_latent_dimension,
            self._generator_number_samples_per_class["number_classes"],
            dense_generator,
            self._get_activation(self._generator_activation_function)
        )

        return generator

    @property
    def dense_generator_model(self):
        """Property that retrieves the dense generator submodel without label conditioning."""
        return self._generator_model_dense

    @property
    def dropout_decay_rate_generator(self) -> float:
        """Property to get the dropout decay rate for the generator."""
        return self._generator_dropout_decay_rate_g

    @property
    def dense_layer_sizes_generator(self) -> List[int]:
        """Property to get the dense layer sizes for the generator."""
        return self._generator_dense_layer_sizes_g

    @dropout_decay_rate_generator.setter
    def dropout_decay_rate_generator(self, dropout_decay_rate_generator: float):
        """Property to set the dropout decay rate for the generator."""
        if not isinstance(dropout_decay_rate_generator, (int, float)) or not (0 <= dropout_decay_rate_generator <= 1):
            raise ValueError(f"Invalid dropout rate: {dropout_decay_rate_generator}. Must be between 0 and 1.")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    @dense_layer_sizes_generator.setter
    def dense_layer_sizes_generator(self, dense_layer_sizes_generator: List[int]):
        """Property to set the dense layer sizes for the generator."""
        if not isinstance(dense_layer_sizes_generator, list) or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_generator):
            raise ValueError(f"Invalid dense layer sizes: {dense_layer_sizes_generator}. Must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator


def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
