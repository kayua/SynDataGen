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

import os
import sys

# Get framework preference from environment variable
FRAMEWORK_ENV = os.environ.get('ML_FRAMEWORK', '').lower()

FRAMEWORK = None

if FRAMEWORK_ENV:
    # Try to load the specified framework
    if FRAMEWORK_ENV == 'tensorflow':
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import Layer as TFLayer
            FRAMEWORK = 'tensorflow'
            print(f"Using framework from ML_FRAMEWORK: {FRAMEWORK}")
        except ImportError:
            print(f"Error: ML_FRAMEWORK set to '{FRAMEWORK_ENV}' but TensorFlow is not installed.")
            sys.exit(-1)
    elif FRAMEWORK_ENV == 'pytorch':
        try:
            import torch
            import torch.nn as nn
            FRAMEWORK = 'pytorch'
            print(f"Using framework from ML_FRAMEWORK: {FRAMEWORK}")
        except ImportError:
            print(f"Error: ML_FRAMEWORK set to '{FRAMEWORK_ENV}' but PyTorch is not installed.")
            sys.exit(-1)
    else:
        print(f"Error: Invalid ML_FRAMEWORK value '{FRAMEWORK_ENV}'. Valid options: 'tensorflow' or 'pytorch'.")
        sys.exit(-1)
else:
    # Auto-detect available framework if no preference is set
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Layer as TFLayer
        FRAMEWORK = 'tensorflow'
    except ImportError:
        pass

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if FRAMEWORK is None:
            FRAMEWORK = 'pytorch'
    except ImportError:
        pass

    if FRAMEWORK is None:
        print("Error: Neither TensorFlow nor PyTorch is installed.")
        sys.exit(-1)
    else:
        print(f"Auto-detected framework: {FRAMEWORK}")



class LeakyReLU:
    """
    Leaky Rectified Linear Unit (LeakyReLU) Activation Function Layer (Framework Agnostic).

    The Leaky ReLU activation function is defined as:

        leaky_relu(x) = x if x > 0
        leaky_relu(x) = negative_slope * x if x <= 0

    It allows a small, non-zero gradient for negative inputs, making it useful for
    overcoming the vanishing gradient problem.

    Attributes
    ----------
        negative_slope : float
            Slope for negative values (default is 0.2).

    Methods
    -------
        forward(neural_network_flow) / call(neural_network_flow)
            Applies the Leaky ReLU activation function to the input tensor and returns the output tensor.

    Example (TensorFlow)
    -------
    >>> import tensorflow as tf
    ...    # Example tensor with shape (batch_size, sequence_length, 8)
    ...    input_tensor = tf.random.uniform((2, 5, 8))
    ...    # Instantiate and apply LeakyReLU
    ...    leaky_relu_layer = LeakyReLU(negative_slope=0.2)
    ...    output_tensor = leaky_relu_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 8)
    ...    print(output_tensor.shape)  # (2, 5, 8)

    Example (PyTorch)
    -------
    >>> import torch
    ...    # Example tensor with shape (batch_size, sequence_length, 8)
    ...    input_tensor = torch.randn(2, 5, 8)
    ...    # Instantiate and apply LeakyReLU
    ...    leaky_relu_layer = LeakyReLU(negative_slope=0.2)
    ...    output_tensor = leaky_relu_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 8)
    ...    print(output_tensor.shape)  # torch.Size([2, 5, 8])
    """

    def __new__(cls, negative_slope=0.2, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            negative_slope (float): Slope for negative values (default is 0.2).
            **kwargs: Additional keyword arguments.
            
        Returns:
            LeakyReLUTF or LeakyReLUPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return LeakyReLUTF(negative_slope=negative_slope, **kwargs)
        elif FRAMEWORK == 'pytorch':
            return LeakyReLUPyTorch(negative_slope=negative_slope, **kwargs)


class LeakyReLUTF(TFLayer):
    """TensorFlow implementation of LeakyReLU."""

    def __init__(self, negative_slope=0.2, **kwargs):
        """
        Initializes the Leaky ReLU activation function layer for TensorFlow.

        Parameters
        ----------
        negative_slope : float
            Slope for negative values (default is 0.2).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(LeakyReLUTF, self).__init__(**kwargs)
        self.negative_slope = negative_slope

    def call(self, neural_network_flow):
        """
        Applies the Leaky ReLU activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input, after applying Leaky ReLU transformation.
        """
        return tf.maximum(self.negative_slope * neural_network_flow, neural_network_flow)

    @staticmethod
    def compute_output_shape(input_shape):
        """
        Computes the output shape, which remains the same as the input shape.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape, identical to input shape.
        """
        return input_shape

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super(LeakyReLUTF, self).get_config()
        config.update({'negative_slope': self.negative_slope})
        return config


class LeakyReLUPyTorch(nn.Module):
    """PyTorch implementation of LeakyReLU."""

    def __init__(self, negative_slope=0.2, **kwargs):
        """
        Initializes the Leaky ReLU activation function layer for PyTorch.

        Parameters
        ----------
        negative_slope : float
            Slope for negative values (default is 0.2).
        **kwargs
            Additional keyword arguments.
        """
        super(LeakyReLUPyTorch, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, neural_network_flow):
        """
        Applies the Leaky ReLU activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : torch.Tensor
            Input tensor with any shape.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input, after applying Leaky ReLU transformation.
        """
        return F.leaky_relu(neural_network_flow, negative_slope=self.negative_slope)

    def extra_repr(self):
        """
        Returns a string representation of the layer parameters.

        Returns
        -------
        str
            String representation showing negative_slope parameter.
        """
        return f'negative_slope={self.negative_slope}'


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
