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
import numpy as np

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
    if FRAMEWORK is None:
        FRAMEWORK = 'pytorch'
except ImportError:
    pass

if FRAMEWORK is None:
    print("Error: Neither TensorFlow nor PyTorch is installed.")
    sys.exit(-1)


class TimeEmbedding:
    """
    This class implements a sinusoidal time embedding layer (Framework Agnostic).
    
    The purpose of this layer is to generate a time-dependent embedding vector, which can be used 
    to encode temporal information. The embedding method is inspired by the positional encoding 
    technique used in transformer models.

    Attributes:
    -----------
        dimension : int
            The dimension of the output embedding. It should be an even number because the embedding is
            split between sine and cosine components.
        _half_dimension : int
            Half of the embedding dimension. Used to calculate the sine and cosine components separately.
        _embedding : Tensor
            Pre-computed scaling factors for each embedding dimension. This is based on the formula
            10000^(2i/d), where 'i' is the index and 'd' is the dimension.

    Methods:
    --------
        __init__(self, dimension, **kwargs)
            Initializes the TimeEmbedding layer, setting the embedding dimension and calculating the scaling factors.

        forward(self, inputs) / call(self, inputs)
            Computes the time embedding for the given inputs using sine and cosine functions. This method
            takes a tensor of inputs representing time steps and returns the corresponding embedding.

    Example (TensorFlow):
    >>>     import tensorflow as tf
    ...     time_emb = TimeEmbedding(dimension=128)
    ...     time_steps = tf.constant([1.0, 2.0, 3.0])
    ...     embeddings = time_emb(time_steps)
    >>>     print(embeddings.shape)  # (3, 128)

    Example (PyTorch):
    >>>     import torch
    ...     time_emb = TimeEmbedding(dimension=128)
    ...     time_steps = torch.tensor([1.0, 2.0, 3.0])
    ...     embeddings = time_emb(time_steps)
    >>>     print(embeddings.shape)  # torch.Size([3, 128])

    """

    def __new__(cls, dimension, **kwargs):
        """
        Factory method to instantiate the appropriate framework-specific implementation.
        
        Args:
            dimension (int): The dimension of the output embedding vector. Must be even.
            **kwargs: Additional keyword arguments.
            
        Returns:
            TimeEmbeddingTF or TimeEmbeddingPyTorch instance.
        """
        if FRAMEWORK == 'tensorflow':
            return TimeEmbeddingTF(dimension, **kwargs)
        elif FRAMEWORK == 'pytorch':
            return TimeEmbeddingPyTorch(dimension, **kwargs)


class TimeEmbeddingTF(TFLayer):
    """TensorFlow implementation of TimeEmbedding."""

    def __init__(self, dimension, **kwargs):
        """
        Initializes the TimeEmbedding layer for TensorFlow.

        Parameters:
        -----------
        dimension : int
            The dimension of the output embedding vector. Must be an even number for the split
            between sine and cosine components.
        **kwargs : dict
            Additional keyword arguments for the base Layer class.
        """
        super().__init__(**kwargs)
        self._dimension = dimension
        self._half_dimension = dimension // 2

        # Compute the scaling factors for each dimension in the embedding.
        self._embedding = np.log(10000) / (self._half_dimension - 1)
        self._embedding = tf.exp(tf.range(self._half_dimension,
                                          dtype=tf.float32) * -self._embedding)

    def call(self, inputs):
        """
        Computes the time embedding for a batch of inputs using sine and cosine functions.

        Parameters:
        -----------
        inputs : Tensor
            A 1D or 2D tensor containing time steps, with shape (batch_size,) or (batch_size, 1).

        Returns:
        --------
        Tensor
            A 2D tensor with shape (batch_size, dimension) containing the computed time embeddings.
        """
        # Cast the input to a float32 tensor.
        inputs = tf.cast(inputs, dtype=tf.float32)

        # Scale the input by the pre-computed embedding factors.
        time_embedding = inputs[:, None] * self._embedding[None, :]

        # Concatenate the sine and cosine values along the last axis to form the final embedding.
        time_embedding = tf.concat([tf.sin(time_embedding),
                                    tf.cos(time_embedding)], axis=-1)
        return time_embedding


class TimeEmbeddingPyTorch(nn.Module):
    """PyTorch implementation of TimeEmbedding."""

    def __init__(self, dimension, **kwargs):
        """
        Initializes the TimeEmbedding layer for PyTorch.

        Parameters:
        -----------
        dimension : int
            The dimension of the output embedding vector. Must be an even number for the split
            between sine and cosine components.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__()
        self._dimension = dimension
        self._half_dimension = dimension // 2

        # Compute the scaling factors for each dimension in the embedding.
        embedding = np.log(10000) / (self._half_dimension - 1)
        embedding = torch.exp(torch.arange(self._half_dimension, dtype=torch.float32) * -embedding)
        
        # Register as a buffer so it moves with the model to GPU/CPU
        self.register_buffer('_embedding', embedding)

    def forward(self, inputs):
        """
        Computes the time embedding for a batch of inputs using sine and cosine functions.

        Parameters:
        -----------
        inputs : Tensor
            A 1D or 2D tensor containing time steps, with shape (batch_size,) or (batch_size, 1).

        Returns:
        --------
        Tensor
            A 2D tensor with shape (batch_size, dimension) containing the computed time embeddings.
        """
        # Ensure inputs is a float tensor
        inputs = inputs.float()
        
        # If inputs is 1D, keep it 1D for indexing; if 2D, squeeze the last dimension
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)

        # Scale the input by the pre-computed embedding factors.
        time_embedding = inputs[:, None] * self._embedding[None, :]

        # Concatenate the sine and cosine values along the last axis to form the final embedding.
        time_embedding = torch.cat([torch.sin(time_embedding),
                                    torch.cos(time_embedding)], dim=-1)
        return time_embedding


# Convenience function to get current framework
def get_framework():
    """Returns the currently active framework ('tensorflow' or 'pytorch')."""
    return FRAMEWORK
