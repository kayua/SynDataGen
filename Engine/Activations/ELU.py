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




class ELU:
    """
    Exponential Linear Unit (ELU) Activation Function Layer (Framework Agnostic).

    The Exponential Linear Unit (ELU) was introduced by Clevert et al. (2016)
    in the paper "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
    (https://arxiv.org/abs/1511.07289). It is defined as:

        elu(x) = x if x > 0 else alpha * (exp(x) - 1)

    ELUs help push the mean activations closer to zero, enabling faster learning
    and reducing the vanishing gradient problem.

    Attributes
    ----------
        alpha : float
            The scaling parameter for negative values (default is 1.0).

    Methods
    -------
        forward(neural_network_flow) / call(neural_network_flow)
            Applies the ELU activation function to the input tensor and returns the output tensor.

    Example (TensorFlow)
    -------
    >>> import tensorflow as tf
    ...    input_tensor = tf.random.uniform((2, 5))
    ...    elu_layer = ELU(alpha=1.0)
    ...    output_tensor = elu_layer(input_tensor)
    ...    print(output_tensor.shape)  # (2, 5)

    Example (PyTorch)
    -------
    >>> import torch
    ...    input_tensor = torch.randn(2, 5)
    ...    elu_layer = ELU(alpha=1.0)
    ...    output_tensor = elu_layer(input_tensor)
    ...    print(output_tensor.shape)  # torch.Size([2, 5])
    """

    def __new__(cls, alpha=1.0, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            alpha (float): The scaling parameter for negative values (default is 1.0).
            **kwargs: Additional keyword arguments.
            
        Returns:
            ELUTF or ELUPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return ELUTF(alpha, **kwargs)
        elif FRAMEWORK == 'pytorch':
            return ELUPyTorch(alpha, **kwargs)


class ELUTF(TFLayer):
    """TensorFlow implementation of ELU."""

    def __init__(self, alpha=1.0, **kwargs):
        """
        Initializes the ELU activation function layer for TensorFlow.

        Parameters
        ----------
        alpha : float, optional
            The scaling parameter for negative values (default is 1.0).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(ELUTF, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, neural_network_flow):
        """
        Applies the ELU activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : tf.Tensor
            Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input, after applying ELU transformation.
        """
        return tf.nn.elu(neural_network_flow) * self.alpha

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


class ELUPyTorch(nn.Module):
    """PyTorch implementation of ELU."""

    def __init__(self, alpha=1.0, **kwargs):
        """
        Initializes the ELU activation function layer for PyTorch.

        Parameters
        ----------
        alpha : float, optional
            The scaling parameter for negative values (default is 1.0).
        **kwargs
            Additional keyword arguments.
        """
        super(ELUPyTorch, self).__init__()
        self.alpha = alpha

    def forward(self, neural_network_flow):
        """
        Applies the ELU activation function to the input tensor.

        Parameters
        ----------
        neural_network_flow : torch.Tensor
            Input tensor with any shape.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input, after applying ELU transformation.
        """
        return F.elu(neural_network_flow, alpha=self.alpha)

    def extra_repr(self):
        """
        Returns a string representation of the layer parameters.

        Returns
        -------
        str
            String representation showing alpha parameter.
        """
        return f'alpha={self.alpha}'


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
