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
import math

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


class GELU:
    """
    Gaussian Error Linear Unit (GELU) Activation Function Layer (Framework Agnostic).

    The Gaussian Error Linear Unit (GELU) is an activation function introduced by Hendrycks and Gimpel (2016)
    in the paper "Gaussian Error Linear Units (GELUs)" (https://arxiv.org/abs/1606.08415). It is widely used
    in deep learning architectures, including Transformer-based models, due to its smooth and adaptive
    non-linearity.

    GELU is a smoother alternative to ReLU and approximates the behavior of dropout by adaptively gating
    the input using a scaled error function.

    Mathematical Definition
    ----------------------
    Given an input tensor `X`, the GELU activation is defined as:

        `gelu(x) = x * P(X <= x)` where `P(X) ~ N(0, 1)`,
        i.e. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.

    This formulation provides a close approximation of the true Gaussian CDF-based activation function.

    Attributes
    ----------
        None

    Methods
    -------
        forward(neural_network_flow) / call(neural_network_flow)
            Applies the GELU activation function to the input tensor and returns the output tensor.

    Example (TensorFlow)
    -------
    >>> import tensorflow as tf
    ...    input_tensor = tf.random.uniform((2, 5))
    ...    gelu_layer = GELU()
    ...    output_tensor = gelu_layer(input_tensor)
    ...    print(output_tensor.shape)  # (2, 5)

    Example (PyTorch)
    -------
    >>> import torch
    ...    input_tensor = torch.randn(2, 5)
    ...    gelu_layer = GELU()
    ...    output_tensor = gelu_layer(input_tensor)
    ...    print(output_tensor.shape)  # torch.Size([2, 5])
    """

    def __new__(cls, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            **kwargs: Additional keyword arguments.
            
        Returns:
            GELUTF or GELUPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return GELUTF(**kwargs)
        elif FRAMEWORK == 'pytorch':
            return GELUPyTorch(**kwargs)


class GELUTF(TFLayer):
    """TensorFlow implementation of GELU."""

    def __init__(self, **kwargs):
        """
        Initializes the Gaussian Error Linear Unit (GELU) layer for TensorFlow.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(GELUTF, self).__init__(**kwargs)

    def call(self, neural_network_flow):
        """
        Applies the GELU activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input, after applying GELU transformation.

        Example
        -------
        >>> input_tensor = tf.random.uniform((2, 5))
        ...     gelu = GELU()
        ...     output = gelu(input_tensor)
        ...     print(output.shape)  # (2, 5)
        """
        return tf.nn.gelu(neural_network_flow)

    def compute_output_shape(self, input_shape):
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


class GELUPyTorch(nn.Module):
    """PyTorch implementation of GELU."""

    def __init__(self, **kwargs):
        """
        Initializes the Gaussian Error Linear Unit (GELU) layer for PyTorch.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        super(GELUPyTorch, self).__init__()

    def forward(self, neural_network_flow):
        """
        Applies the GELU activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : torch.Tensor
            Input tensor with any shape.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input, after applying GELU transformation.

        Example
        -------
        >>> input_tensor = torch.randn(2, 5)
        ...     gelu = GELU()
        ...     output = gelu(input_tensor)
        ...     print(output.shape)  # torch.Size([2, 5])
        """
        return F.gelu(neural_network_flow)


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
