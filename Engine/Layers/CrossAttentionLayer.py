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
    from tensorflow.keras.layers import Dense, Layer as TFLayer
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


class CrossAttentionBlock:
    """
    A custom layer that implements a Cross-Attention mechanism (Framework Agnostic).

    This layer computes cross-attention between two sets of input sequences:
    queries and key-value pairs. The cross-attention mechanism computes attention
    scores between the queries and keys and uses the resulting attention weights
    to perform weighted aggregation of the values. The output of the attention is
    projected into a desired number of units. It also incorporates a residual
    connection between the input and the attention output.

    The CrossAttentionBlock layer is inspired by the self-attention mechanism described
    in the paper "Attention is All You Need" (Vaswani et al., 2017), but this version
    operates with queries and key-value pairs from separate inputs, making it suitable
    for tasks such as cross-modal learning or multi-view attention.

    References:
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser,
        Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information
        Processing Systems, 30. URL: https://arxiv.org/abs/1706.03762

    Mathematical Definition:
        Let Q represent the query matrix of shape (batch_size, seq_len, num_channels),
        K represent the key matrix, and V represent the value matrix, both of shape (batch_size, seq_len).

        The attention mechanism can be described as:

        Attention Scores = (Q ⋅ Kᵀ) / √d_k
        where d_k is the dimension of the key vectors (i.e., the number of units in this case).

        Then, the attention weights are computed as:

        Attention Weights = softmax(Attention Scores)

        The output of the attention mechanism is:

        Attention Output = Attention Weights ⋅ V

        Finally, the output is projected by applying a linear transformation and adding the input values as a residual:

        Final Output = Input + Projection(Attention Output)

    Attributes:
        @units: Integer representing the number of output units for each attention head.
        @query_weights: Dense/Linear layer for the transformation of the query inputs.
        @key_weights: Dense/Linear layer for the transformation of the key inputs.
        @value_weights: Dense/Linear layer for the transformation of the value inputs.
        @projection_weights: Dense/Linear layer for projecting the attention output.

    Methods:
        forward(input_values) / call(input_values): Computes the attention output and applies the residual connection.

    Example (TensorFlow):
    >>>     import tensorflow as tf
    ...     query_inputs = tf.random.normal((2, 5, 3))
    ...     key_value_inputs = tf.random.normal((2, 5))
    ...     cross_attention_block = CrossAttentionBlock(units=4)
    ...     output = cross_attention_block([query_inputs, key_value_inputs])
    >>>     print(output.shape)  # Expected output shape: (2, 5, 4)

    Example (PyTorch):
    >>>     import torch
    ...     query_inputs = torch.randn(2, 5, 3)
    ...     key_value_inputs = torch.randn(2, 5)
    ...     cross_attention_block = CrossAttentionBlock(units=4)
    ...     output = cross_attention_block([query_inputs, key_value_inputs])
    >>>     print(output.shape)  # Expected output shape: torch.Size([2, 5, 4])

    """

    def __new__(cls, units, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            units (int): The number of output units for the attention block.
            **kwargs: Additional keyword arguments.
            
        Returns:
            CrossAttentionBlockTF or CrossAttentionBlockPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return CrossAttentionBlockTF(units, **kwargs)
        elif FRAMEWORK == 'pytorch':
            return CrossAttentionBlockPyTorch(units, **kwargs)


class CrossAttentionBlockTF(TFLayer):
    """TensorFlow implementation of CrossAttentionBlock."""

    def __init__(self, units, **kwargs):
        """
        Initializes the CrossAttentionBlock layer for TensorFlow.

        Args:
            units (int): The number of output units for the attention block.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.units = units

        # Initialize the weight matrices for query, key, value, and output projection
        self.query_weights = Dense(units)
        self.key_weights = Dense(units)
        self.value_weights = Dense(units)
        self.projection_weights = Dense(units)

    def call(self, input_values):
        """
        Performs the forward pass of the CrossAttentionBlock layer.

        Args:
            input_values (list): A list containing two tensors:
                - query_inputs (Tensor): The query tensor of shape (batch_size, seq_len, num_channels).
                - key_value_inputs (Tensor): The key-value tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: The resulting tensor of shape (batch_size, seq_len, units),
                    which is the sum of the original query inputs and the attention output.
        """
        # Extract query inputs and key-value inputs from the input values
        query_inputs, key_value_inputs = input_values

        # Get the dimensions for batch size, sequence length, and number of channels
        number_channels = tf.shape(query_inputs)[2]

        # Expand key_value_inputs to match the shape (batch_size, seq_len, num_channels)
        key_value_inputs = tf.tile(tf.expand_dims(key_value_inputs, axis=-1), [1, 1, number_channels])

        # Calculate scaling factor for attention scores
        scale = tf.cast(self.units, tf.float32) ** -0.5

        # Apply linear projections to queries, keys, and values
        query = self.query_weights(query_inputs)
        key = self.key_weights(key_value_inputs)
        value = self.value_weights(key_value_inputs)

        # Compute attention scores
        attention_scores = tf.matmul(query, key, transpose_b=True) * scale

        # Apply softmax to obtain attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Compute attention output by applying attention weights to values
        attention_output = tf.matmul(attention_weights, value)

        # Project the attention output to the desired dimensionality
        attention_output = self.projection_weights(attention_output)

        # Apply residual connection by adding the input query values to the attention output
        return query_inputs + attention_output


class CrossAttentionBlockPyTorch(nn.Module):
    """PyTorch implementation of CrossAttentionBlock."""

    def __init__(self, units, **kwargs):
        """
        Initializes the CrossAttentionBlock layer for PyTorch.

        Args:
            units (int): The number of output units for the attention block.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.units = units

        # Linear layers will be initialized dynamically in the first forward pass
        self.query_weights = None
        self.key_weights = None
        self.value_weights = None
        self.projection_weights = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """
        Initializes the linear layers based on the input dimension.
        
        Args:
            input_dim (int): The dimension of the input features.
        """
        self.query_weights = nn.Linear(input_dim, self.units)
        self.key_weights = nn.Linear(input_dim, self.units)
        self.value_weights = nn.Linear(input_dim, self.units)
        self.projection_weights = nn.Linear(self.units, input_dim)
        self._initialized = True

    def forward(self, input_values):
        """
        Performs the forward pass of the CrossAttentionBlock layer.

        Args:
            input_values (list or tuple): A list/tuple containing two tensors:
                - query_inputs (Tensor): The query tensor of shape (batch_size, seq_len, num_channels).
                - key_value_inputs (Tensor): The key-value tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: The resulting tensor of shape (batch_size, seq_len, num_channels),
                    which is the sum of the original query inputs and the attention output.
        """
        # Extract query inputs and key-value inputs from the input values
        query_inputs, key_value_inputs = input_values

        # Initialize layers on first forward pass
        if not self._initialized:
            input_dim = query_inputs.shape[-1]
            self._initialize_layers(input_dim)

        # Get the dimensions
        batch_size, seq_len, number_channels = query_inputs.shape

        # Expand key_value_inputs to match the shape (batch_size, seq_len, num_channels)
        key_value_inputs = key_value_inputs.unsqueeze(-1).expand(batch_size, seq_len, number_channels)

        # Calculate scaling factor for attention scores
        scale = self.units ** -0.5

        # Apply linear projections to queries, keys, and values
        query = self.query_weights(query_inputs)
        key = self.key_weights(key_value_inputs)
        value = self.value_weights(key_value_inputs)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply softmax to obtain attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute attention output by applying attention weights to values
        attention_output = torch.matmul(attention_weights, value)

        # Project the attention output to the desired dimensionality
        attention_output = self.projection_weights(attention_output)

        # Apply residual connection by adding the input query values to the attention output
        return query_inputs + attention_output


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
