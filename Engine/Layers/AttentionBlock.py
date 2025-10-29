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

DEFAULT_GROUP_NORMALIZATION = 1


class AttentionBlock:
    """
    AttentionBlock (Framework Agnostic)

    Implements a scaled dot-product attention mechanism for use in deep learning models.
    The block includes query, key, and value projections, followed by a final projection
    layer. This implementation automatically adapts to either TensorFlow or PyTorch.

    This block is inspired by the attention mechanism described in the paper:

    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser, Ł., & Polosukhin, I.
    (2017). Attention is all you need. In Advances in neural information processing systems (Vol. 30).

    Attributes:
        @units (int):
            The number of output units for the dense layers in the attention mechanism.
        @groups (int):
            The number of groups for normalization. Defaults to `DEFAULT_GROUP_NORMALIZATION`.

    Methods:
        @forward(inputs) / @call(inputs):
            Computes the forward pass for the attention block. This method calculates the attention
            scores, applies them to the input, and returns the augmented input after normalization.

    Example:
        >>> attention_block = AttentionBlock(units=64, groups=8)
        >>> output = attention_block(inputs)
    """

    def __new__(cls, units, groups=DEFAULT_GROUP_NORMALIZATION, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            units (int): Number of units for the dense layers in the attention mechanism.
            groups (int, optional): Number of groups for normalization.
            **kwargs: Additional keyword arguments.
            
        Returns:
            AttentionBlockTF or AttentionBlockPyTorch instance depending on available framework.
        """
        if FRAMEWORK == 'tensorflow':
            return AttentionBlockTF(units, groups, **kwargs)
        elif FRAMEWORK == 'pytorch':
            return AttentionBlockPyTorch(units, groups, **kwargs)


class AttentionBlockTF(TFLayer):
    """TensorFlow implementation of AttentionBlock."""

    def __init__(self, units, groups=DEFAULT_GROUP_NORMALIZATION, **kwargs):
        """
        Initializes the AttentionBlock for TensorFlow.

        Args:
            units (int): Number of units for the dense layers.
            groups (int, optional): Number of groups for normalization.
            **kwargs: Additional keyword arguments for the parent Layer class.
        """
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        # Define dense layers for query, key, value, and final projection
        self.query_weights = Dense(units)
        self.key_weights = Dense(units)
        self.value_weights = Dense(units)
        self.projection_weights = Dense(units)

    def call(self, inputs):
        """
        Performs the forward pass of the attention block.

        Args:
            inputs (Tensor): The input tensor of shape (batch_size, height, embedding_dim).

        Returns:
            Tensor: The output tensor after applying attention mechanism and projection.
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        # Compute query, key, and value projections
        query = self.query_weights(inputs)
        key = self.key_weights(inputs)
        value = self.value_weights(inputs)

        # Compute attention scores using scaled dot-product attention
        attention_score = tf.einsum("bhc,bHc->bhH", query, key) * scale
        attention_score = tf.reshape(attention_score, [batch_size, height, height])

        # Apply softmax to obtain attention weights
        attention_score = tf.nn.softmax(attention_score, -1)
        attention_score = tf.reshape(attention_score, [batch_size, height, height])

        # Apply attention weights to the value tensor
        projection = tf.einsum("bhH,bHc->bhc", attention_score, value)
        projection = self.projection_weights(projection)

        # Add the original input to the projection to form the output
        return inputs + projection


class AttentionBlockPyTorch(nn.Module):
    """PyTorch implementation of AttentionBlock."""

    def __init__(self, units, groups=DEFAULT_GROUP_NORMALIZATION, **kwargs):
        """
        Initializes the AttentionBlock for PyTorch.

        Args:
            units (int): Number of units for the linear layers.
            groups (int, optional): Number of groups for normalization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.units = units
        self.groups = groups

        # Define linear layers for query, key, value, and final projection
        self.query_weights = nn.Linear(units, units)
        self.key_weights = nn.Linear(units, units)
        self.value_weights = nn.Linear(units, units)
        self.projection_weights = nn.Linear(units, units)

    def forward(self, inputs):
        """
        Performs the forward pass of the attention block.

        Args:
            inputs (Tensor): The input tensor of shape (batch_size, height, embedding_dim).

        Returns:
            Tensor: The output tensor after applying attention mechanism and projection.
        """
        batch_size, height, _ = inputs.shape
        scale = self.units ** (-0.5)

        # Compute query, key, and value projections
        query = self.query_weights(inputs)
        key = self.key_weights(inputs)
        value = self.value_weights(inputs)

        # Compute attention scores using scaled dot-product attention
        attention_score = torch.einsum("bhc,bHc->bhH", query, key) * scale
        attention_score = attention_score.reshape(batch_size, height, height)

        # Apply softmax to obtain attention weights
        attention_score = torch.softmax(attention_score, dim=-1)
        attention_score = attention_score.reshape(batch_size, height, height)

        # Apply attention weights to the value tensor
        projection = torch.einsum("bhH,bHc->bhc", attention_score, value)
        projection = self.projection_weights(projection)

        # Add the original input to the projection to form the output
        return inputs + projection
