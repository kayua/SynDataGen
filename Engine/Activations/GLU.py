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


class GLU:
    """
    Gated Linear Unit (GLU) Layer (Framework Agnostic).

    The Gated Linear Unit (GLU) is an activation function introduced by Dauphin et al. (2017)
    in the paper "Language Modeling with Gated Convolutional Networks" (https://arxiv.org/abs/1612.08083).
    It applies a gating mechanism to the input tensor, where the input is split into two halves:
    one half is transformed directly, while the other half is passed through a sigmoid gate.
    The element-wise product of these two halves forms the final output.

    This mechanism helps the network selectively retain or suppress information, improving
    the modeling of complex relationships in the data. GLUs have shown effectiveness in both
    convolutional and sequential models.

    Mathematical Definition
    ----------------------
    Given an input tensor `X` with shape (..., 2d), the GLU activation is defined as:

        GLU(X) = X1 ⊗ σ(X2)

    Where:
        - X1 ∈ ℝ^{..., d} (first half of the last dimension of X)
        - X2 ∈ ℝ^{..., d} (second half of the last dimension of X)
        - σ denotes the sigmoid activation function
        - ⊗ denotes element-wise multiplication

    The input tensor must have an even number of channels in the last dimension.

    Attributes
    ----------
        None

    Methods
    -------
        forward(neural_network_flow) / call(neural_network_flow)
            Applies the GLU activation function to the input tensor and returns the output tensor.

    Example (TensorFlow)
    -------
    >>> import tensorflow as tf
    ...    # Example tensor with shape (batch_size, sequence_length, 8) — divisible by 2
    ...    input_tensor = tf.random.uniform((2, 5, 8))
    ...    # Instantiate and apply GLU
    ...    glu_layer = GLU()
    ...    output_tensor = glu_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 4)
    ...    print(output_tensor.shape)  # (2, 5, 4)

    Example (PyTorch)
    -------
    >>> import torch
    ...    # Example tensor with shape (batch_size, sequence_length, 8) — divisible by 2
    ...    input_tensor = torch.randn(2, 5, 8)
    ...    # Instantiate and apply GLU
    ...    glu_layer = GLU()
    ...    output_tensor = glu_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 4)
    ...    print(output_tensor.shape)  # torch.Size([2, 5, 4])
    """

    def __new__(cls, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            **kwargs: Additional keyword arguments.
            
        Returns:
            GLUTF or GLUPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return GLUTF(**kwargs)
        elif FRAMEWORK == 'pytorch':
            return GLUPyTorch(**kwargs)


class GLUTF(TFLayer):
    """TensorFlow implementation of GLU."""

    def __init__(self, **kwargs):
        """
        Initializes the Gated Linear Unit (GLU) layer for TensorFlow.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(GLUTF, self).__init__(**kwargs)

    def call(self, neural_network_flow):
        """
        Applies the Gated Linear Unit (GLU) activation function to the input tensor.

        The input tensor is split along the last dimension into two equal parts.
        The first half represents the linear transformation, and the second half
        passes through a sigmoid gate. The final output is the element-wise product
        of these two halves.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor with shape (..., 2d). The last dimension must be even.

        Returns
        -------
        tf.Tensor
            Output tensor with shape (..., d), after applying the GLU transformation.

        Raises
        ------
        ValueError
            If the last dimension of the input tensor is not divisible by 2.

        Example
        -------
        >>> input_tensor = tf.random.uniform((2, 5, 8))
        ...     glu = GLU()
        ...     output = glu(input_tensor)
        ...     print(output.shape)  # (2, 5, 4)
        """
        # Check if last dimension is divisible by 2
        if neural_network_flow.shape[-1] % 2 != 0:
            raise ValueError(f"Last dimension of input tensor must be divisible by 2, but got {neural_network_flow.shape[-1]}")
        
        a, b = tf.split(neural_network_flow, 2, axis=-1)
        return a * tf.nn.sigmoid(b)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape after applying GLU.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape, where the last dimension is halved.

        Raises
        ------
        ValueError
            If the last dimension of the input shape is not divisible by 2.
        """
        if input_shape[-1] % 2 != 0:
            raise ValueError(f"Last dimension of input shape must be divisible by 2, but got {input_shape[-1]}")
        
        return input_shape[:-1] + (input_shape[-1] // 2,)


class GLUPyTorch(nn.Module):
    """PyTorch implementation of GLU."""

    def __init__(self, **kwargs):
        """
        Initializes the Gated Linear Unit (GLU) layer for PyTorch.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        super(GLUPyTorch, self).__init__()

    def forward(self, neural_network_flow):
        """
        Applies the Gated Linear Unit (GLU) activation function to the input tensor.

        The input tensor is split along the last dimension into two equal parts.
        The first half represents the linear transformation, and the second half
        passes through a sigmoid gate. The final output is the element-wise product
        of these two halves.

        Parameters
        ----------
        neural_network_flow : torch.Tensor
            Input tensor with shape (..., 2d). The last dimension must be even.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (..., d), after applying the GLU transformation.

        Raises
        ------
        ValueError
            If the last dimension of the input tensor is not divisible by 2.

        Example
        -------
        >>> input_tensor = torch.randn(2, 5, 8)
        ...     glu = GLU()
        ...     output = glu(input_tensor)
        ...     print(output.shape)  # torch.Size([2, 5, 4])
        """
        # Check if last dimension is divisible by 2
        if neural_network_flow.size(-1) % 2 != 0:
            raise ValueError(f"Last dimension of input tensor must be divisible by 2, but got {neural_network_flow.size(-1)}")
        
        # Split along the last dimension
        a, b = torch.chunk(neural_network_flow, 2, dim=-1)
        return a * torch.sigmoid(b)


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
