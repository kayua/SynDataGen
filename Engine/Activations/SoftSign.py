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



class SoftSign:
    """
    Softsign Activation Function Layer (Framework Agnostic).

    Softsign is defined as: `softsign(x) = x / (abs(x) + 1)`.

    Attributes
    ----------
    None

    Methods
    -------
    forward(neural_network_flow) / call(neural_network_flow)
        Applies the Softsign activation function to the input tensor and returns the output tensor.

    Example (TensorFlow)
    -------
    >>> import tensorflow as tf
    ...    # Example tensor with shape (batch_size, sequence_length, 8)
    ...    input_tensor = tf.random.uniform((2, 5, 8))
    ...    # Instantiate and apply Softsign
    ...    softsign_layer = SoftSign()
    ...    output_tensor = softsign_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 8)
    ...    print(output_tensor.shape)  # (2, 5, 8)

    Example (PyTorch)
    -------
    >>> import torch
    ...    # Example tensor with shape (batch_size, sequence_length, 8)
    ...    input_tensor = torch.randn(2, 5, 8)
    ...    # Instantiate and apply Softsign
    ...    softsign_layer = SoftSign()
    ...    output_tensor = softsign_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 8)
    ...    print(output_tensor.shape)  # torch.Size([2, 5, 8])
    """

    def __new__(cls, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            **kwargs: Additional keyword arguments.
            
        Returns:
            SoftSignTF or SoftSignPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return SoftSignTF(**kwargs)
        elif FRAMEWORK == 'pytorch':
            return SoftSignPyTorch(**kwargs)


class SoftSignTF(TFLayer):
    """TensorFlow implementation of SoftSign."""

    def __init__(self, **kwargs):
        """
        Initializes the Softsign activation function layer for TensorFlow.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(SoftSignTF, self).__init__(**kwargs)

    def call(self, neural_network_flow):
        """
        Applies the Softsign activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input, after applying Softsign transformation.
        """
        return tf.nn.softsign(neural_network_flow)

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


class SoftSignPyTorch(nn.Module):
    """PyTorch implementation of SoftSign."""

    def __init__(self, **kwargs):
        """
        Initializes the Softsign activation function layer for PyTorch.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        super(SoftSignPyTorch, self).__init__()

    def forward(self, neural_network_flow):
        """
        Applies the Softsign activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : torch.Tensor
            Input tensor with any shape.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input, after applying Softsign transformation.
        """
        return F.softsign(neural_network_flow)

    @staticmethod
    def extra_repr():
        """
        Returns a string representation of the layer.

        Returns
        -------
        str
            String representation of the layer.
        """
        return 'softsign'


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
