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

# Detect available framework
FRAMEWORK = None
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


class Swish:
    """
    Swish Activation Function Layer (Framework Agnostic).

    The Swish activation function was introduced by Ramachandran et al. (2017)
    in the paper "Searching for Activation Functions" (https://arxiv.org/abs/1710.05941).
    It is defined as:

        swish(x) = x * sigmoid(x)

    Swish is a smooth, non-monotonic function that has been shown to outperform
    ReLU in some deep learning architectures.

    Attributes
    ----------
        None

    Methods
    -------
        forward(neural_network_flow) / call(neural_network_flow)
            Applies the Swish activation function to the input tensor and returns the output tensor.

    Example (TensorFlow)
    -------
    >>> import tensorflow as tf
    ...    # Example tensor with shape (batch_size, feature_dim)
    ...    input_tensor = tf.random.uniform((2, 5))
    ...    # Instantiate and apply Swish
    ...    swish_layer = Swish()
    ...    output_tensor = swish_layer(input_tensor)
    ...    # Output shape (batch_size, feature_dim)
    ...    print(output_tensor.shape)  # (2, 5)

    Example (PyTorch)
    -------
    >>> import torch
    ...    # Example tensor with shape (batch_size, feature_dim)
    ...    input_tensor = torch.randn(2, 5)
    ...    # Instantiate and apply Swish
    ...    swish_layer = Swish()
    ...    output_tensor = swish_layer(input_tensor)
    ...    # Output shape (batch_size, feature_dim)
    ...    print(output_tensor.shape)  # torch.Size([2, 5])
    """

    def __new__(cls, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            **kwargs: Additional keyword arguments.
            
        Returns:
            SwishTF or SwishPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return SwishTF(**kwargs)
        elif FRAMEWORK == 'pytorch':
            return SwishPyTorch(**kwargs)


class SwishTF(TFLayer):
    """TensorFlow implementation of Swish."""

    def __init__(self, **kwargs):
        """
        Initializes the Swish activation function layer for TensorFlow.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(SwishTF, self).__init__(**kwargs)

    def call(self, neural_network_flow):
        """
        Applies the Swish activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input, after applying Swish transformation.

        Example
        -------
        >>> input_tensor = tf.random.uniform((2, 5))
        ...     swish = Swish()
        ...     output = swish(input_tensor)
        ...     print(output.shape)  # (2, 5)
        """
        return neural_network_flow * tf.nn.sigmoid(neural_network_flow)

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


class SwishPyTorch(nn.Module):
    """PyTorch implementation of Swish."""

    def __init__(self, **kwargs):
        """
        Initializes the Swish activation function layer for PyTorch.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        super(SwishPyTorch, self).__init__()

    @staticmethod
    def forward(neural_network_flow):
        """
        Applies the Swish activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : torch.Tensor
            Input tensor with any shape.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input, after applying Swish transformation.

        Example
        -------
        >>> input_tensor = torch.randn(2, 5)
        ...     swish = Swish()
        ...     output = swish(input_tensor)
        ...     print(output.shape)  # torch.Size([2, 5])
        """
        return neural_network_flow * torch.sigmoid(neural_network_flow)

    @staticmethod
    def extra_repr():
        """
        Returns a string representation of the layer.

        Returns
        -------
        str
            String representation of the layer.
        """
        return 'swish'


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
