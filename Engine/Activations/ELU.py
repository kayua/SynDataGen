#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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

try:
    import sys

    import tensorflow

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class ELU(Layer):
    """
    Exponential Linear Unit (ELU) Activation Function Layer.

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
        call(neural_network_flow: tf.Tensor) -> tf.Tensor
            Applies the ELU activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow
    ...    # Example tensor with shape (batch_size, feature_dim)
    ...    input_tensor = tensorflow.random.uniform((2, 5))
    ...    # Instantiate and apply ELU
    ...    elu_layer = ELU(alpha=1.0)
    ...    output_tensor = elu_layer(input_tensor)
    ...    # Output shape (batch_size, feature_dim)
    ...    print(output_tensor.shape)
    >>>
    """

    def __init__(self, alpha=1.0, **kwargs):
        """
        Initializes the ELU activation function layer.

        Parameters
        ----------
        alpha : float, optional
            The scaling parameter for negative values (default is 1.0).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(ELU, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
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

        Example
        -------
        >>> input_tensor = tensorflow.random.uniform((2, 5))
        ...     elu = ELU(alpha=1.0)
        ...     output = elu(input_tensor)
        ...     print(output.shape)
        >>>     (2, 5)
        """
        return tensorflow.nn.elu(neural_network_flow) * self.alpha

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
