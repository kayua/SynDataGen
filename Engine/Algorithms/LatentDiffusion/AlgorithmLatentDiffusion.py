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
    import os
    import sys
    import json
    import numpy
    from typing import Any

except ImportError as error:
    print(error)
    sys.exit(-1)


class LatentDiffusionAlgorithm:
    """
    A framework-agnostic implementation of a diffusion process using UNet architectures 
    for generating synthetic data. Supports both TensorFlow and PyTorch.
    
    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters to adapt to different tasks.

    Attributes:
        @framework (str):
            Framework to use: 'tensorflow' or 'pytorch'.
        @ema (float):
            Exponential moving average (EMA) decay rate for stabilizing training updates.
        @margin (float):
            Margin parameter used for loss computation or regularization purposes.
        @gdf_util:
            Utility object for Gaussian diffusion functions, handling noise scheduling and diffusion-related operations.
        @time_steps (int):
            Number of time steps used in the diffusion process.
        @train_stage (str):
            Defines the current training stage ('all', 'diffusion', etc.), determining whether only specific components are updated.
        @network (Model):
            Primary UNet model responsible for the diffusion process.
        @second_unet_model (Model):
            Secondary UNet model used for EMA-based weight updates to enhance training stability.
        @embedding_dimension (int):
            Dimensionality of the latent space used for encoding data.
        @encoder_model_data (Model):
            Encoder model responsible for feature extraction from input data.
        @decoder_model_data (Model):
            Decoder model used to reconstruct data from encoded representations.
        @optimizer_diffusion (Optimizer):
            Optimizer used for training the diffusion model.
        @optimizer_autoencoder (Optimizer):
            Optimizer responsible for training the autoencoder components.

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
        Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/2006.11239

    Example:
        >>> # TensorFlow example
        >>> diffusion_model = LatentDiffusionAlgorithm(
        ...     framework='tensorflow',
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     encoder_model_image=encoder,
        ...     decoder_model_image=decoder,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=tf.keras.optimizers.Adam(learning_rate=1e-4),
        ...     optimizer_diffusion=tf.keras.optimizers.Adam(learning_rate=2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     embedding_dimension=128,
        ...     train_stage='all'
        ... )
        
        >>> # PyTorch example
        >>> diffusion_model = LatentDiffusionAlgorithm(
        ...     framework='pytorch',
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     encoder_model_image=encoder,
        ...     decoder_model_image=decoder,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=torch.optim.Adam(params, lr=1e-4),
        ...     optimizer_diffusion=torch.optim.Adam(params, lr=2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     embedding_dimension=128,
        ...     train_stage='all'
        ... )
    """

    def __init__(self, 
                 framework,
                 first_unet_model,
                 second_unet_model,
                 encoder_model_image,
                 decoder_model_image,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 embedding_dimension,
                 train_stage='all'):
        """
        Initializes the LatentDiffusionAlgorithm with provided sub-models, optimizers, and hyperparameters.

        Args:
            @framework (str):
                Framework to use: 'tensorflow' or 'pytorch'.
            @first_unet_model (Model):
                Primary UNet model for diffusion-based generation.
            @second_unet_model (Model):
                Secondary UNet model for maintaining EMA-based weight updates.
            @encoder_model_image (Model):
                Encoder model used to extract meaningful feature representations.
            @decoder_model_image (Model):
                Decoder model reconstructing data from encoded embeddings.
            @gdf_util:
                Utility object responsible for Gaussian diffusion operations.
            @optimizer_autoencoder (Optimizer):
                Optimizer handling the training of the encoder-decoder network.
            @optimizer_diffusion (Optimizer):
                Optimizer applied to the diffusion process.
            @time_steps (int):
                Number of discrete time steps for the diffusion process.
            @ema (float):
                Exponential moving average decay factor.
            @margin (float):
                Margin value used in loss calculations or regularization.
            @embedding_dimension (int):
                Dimensionality of the embedding space.
            @train_stage (str, optional):
                Current training stage ('all', 'diffusion', etc.), defaulting to 'all'.

        Raises:
            ValueError:
                If framework is not 'tensorflow' or 'pytorch'.
                If time_steps is <= 0.
                If ema is not within the (0,1) range.
                If embedding_dimension is <= 0.
        """
        
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")
            
        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer.")
            
        if not isinstance(ema, (int, float)) or not (0 < ema < 1):
            raise ValueError("ema must be a float between 0 and 1.")
            
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")

        self._framework = framework
        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._second_unet_model = second_unet_model
        self._embedding_dimension = embedding_dimension
        self._encoder_model_data = encoder_model_image
        self._decoder_model_data = decoder_model_image
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            self.tf = tf
            self._device = None  # TensorFlow handles device placement automatically
            
        else:  # pytorch
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move models to device
            self._network.to(self._device)
            self._second_unet_model.to(self._device)
            self._encoder_model_data.to(self._device)
            self._decoder_model_data.to(self._device)

    def set_stage_training(self, training_stage):
        """
        Sets the current training stage.

        Args:
            training_stage (str): New training stage ('all', 'diffusion', etc.).
        """
        self._train_stage = training_stage

    def train_step(self, data):
        """
        Performs a single training step.

        Args:
            data (tuple): A tuple containing input data and labels.

        Returns:
            dict: A dictionary with the computed loss for diffusion.
        """
        raw_data, label = data
        loss_diffusion = self.train_diffusion_model(raw_data, label)
        self.update_ema_weights()
        return {"Diffusion_loss": loss_diffusion if loss_diffusion is not None else 0}

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        Args:
            data: Input data embeddings (e.g., image or text embeddings).
            ground_truth: Corresponding class labels or conditioning embeddings.

        Returns:
            The computed loss for this training step.
        """
        if self._framework == 'tensorflow':
            return self._train_diffusion_model_tensorflow(data, ground_truth)
        else:
            return self._train_diffusion_model_pytorch(data, ground_truth)

    def _train_diffusion_model_tensorflow(self, data, ground_truth):
        """TensorFlow implementation of train_diffusion_model."""
        embedding_label = ground_truth
        embedding_data_expanded = data
        batch_size = self.tf.shape(data)[0]

        random_time_steps = self.tf.random.uniform(
            minval=0,
            maxval=self._time_steps,
            shape=(batch_size,),
            dtype=self.tf.int32
        )

        with self.tf.GradientTape() as tape:
            random_noise = self.tf.random.normal(
                shape=self.tf.shape(embedding_data_expanded),
                dtype=embedding_data_expanded.dtype
            )

            embedding_with_noise = self._gdf_util.q_sample(
                embedding_data_expanded,
                random_time_steps,
                random_noise
            )

            predicted_noise = self._network(
                [embedding_with_noise, random_time_steps, embedding_label],
                training=True
            )

            loss_diffusion = self.tf.reduce_mean(self.tf.square(random_noise - predicted_noise))

        gradients = tape.gradient(loss_diffusion, self._network.trainable_weights)
        self._optimizer_diffusion.apply_gradients(zip(gradients, self._network.trainable_weights))

        return loss_diffusion

    def _train_diffusion_model_pytorch(self, data, ground_truth):
        """PyTorch implementation of train_diffusion_model."""
        data = data.to(self._device)
        ground_truth = ground_truth.to(self._device)
        
        embedding_label = ground_truth
        embedding_data_expanded = data
        batch_size = data.shape[0]

        random_time_steps = self.torch.randint(
            0,
            self._time_steps,
            (batch_size,),
            device=self._device,
            dtype=self.torch.long
        )

        self._optimizer_diffusion.zero_grad()
        
        random_noise = self.torch.randn_like(embedding_data_expanded)

        embedding_with_noise = self._gdf_util.q_sample(
            embedding_data_expanded,
            random_time_steps,
            random_noise
        )

        predicted_noise = self._network(
            embedding_with_noise,
            random_time_steps,
            embedding_label
        )

        loss_diffusion = self.nn.functional.mse_loss(random_noise, predicted_noise)

        loss_diffusion.backward()
        self._optimizer_diffusion.step()

        return loss_diffusion.item()

    def update_ema_weights(self):
        """
        Updates the weights of the second UNet model using exponential moving average.
        """
        if self._framework == 'tensorflow':
            self._update_ema_weights_tensorflow()
        else:
            self._update_ema_weights_pytorch()

    def _update_ema_weights_tensorflow(self):
        """TensorFlow implementation of update_ema_weights."""
        for weight, ema_weight in zip(self._network.weights, self._second_unet_model.weights):
            ema_weight.assign(self._ema * ema_weight + (1 - self._ema) * weight)

    def _update_ema_weights_pytorch(self):
        """PyTorch implementation of update_ema_weights."""
        with self.torch.no_grad():
            for param, ema_param in zip(self._network.parameters(), self._second_unet_model.parameters()):
                ema_param.data.mul_(self._ema).add_(param.data, alpha=1 - self._ema)

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process.

        Args:
            labels: Class labels used to condition the generated data.
            batch_size (int): Number of data samples to generate in a single batch.

        Returns:
            numpy.ndarray: Generated synthetic data samples.
        """
        if self._framework == 'tensorflow':
            return self._generate_data_tensorflow(labels, batch_size)
        else:
            return self._generate_data_pytorch(labels, batch_size)

    def _generate_data_tensorflow(self, labels, batch_size):
        """TensorFlow implementation of generate_data."""
        embedding_diffusion = self.tf.random.normal(
            shape=(labels.shape[0], self._embedding_dimension, 1),
            dtype=self.tf.float32
        )

        labels_vector = self.tf.expand_dims(labels, axis=-1)

        for time_step in reversed(range(0, self._time_steps)):
            array_time = self.tf.cast(
                self.tf.fill([labels_vector.shape[0]], time_step),
                dtype=self.tf.int32
            )

            predicted_noise = self._network.predict(
                [embedding_diffusion, array_time, labels_vector],
                verbose=0,
                batch_size=batch_size
            )

            embedding_diffusion = self._gdf_util.p_sample(
                predicted_noise[0],
                embedding_diffusion,
                array_time,
                clip_denoised=True
            )

        generated_data = self._decoder_model_data.predict(
            [embedding_diffusion, labels_vector],
            verbose=0,
            batch_size=batch_size // 4
        )

        return generated_data

    def _generate_data_pytorch(self, labels, batch_size):
        """PyTorch implementation of generate_data."""
        self._network.eval()
        self._decoder_model_data.eval()
        
        with self.torch.no_grad():
            if isinstance(labels, numpy.ndarray):
                labels = self.torch.tensor(labels, dtype=self.torch.float32, device=self._device)
            
            embedding_diffusion = self.torch.randn(
                labels.shape[0],
                self._embedding_dimension,
                1,
                device=self._device,
                dtype=self.torch.float32
            )

            labels_vector = labels.unsqueeze(-1)

            for time_step in reversed(range(0, self._time_steps)):
                array_time = self.torch.full(
                    (labels_vector.shape[0],),
                    time_step,
                    device=self._device,
                    dtype=self.torch.long
                )

                predicted_noise = self._network(
                    embedding_diffusion,
                    array_time,
                    labels_vector
                )

                embedding_diffusion = self._gdf_util.p_sample(
                    predicted_noise,
                    embedding_diffusion,
                    array_time,
                    clip_denoised=True
                )

            generated_data = self._decoder_model_data(embedding_diffusion, labels_vector)

        self._network.train()
        self._decoder_model_data.train()
        
        return generated_data.cpu().numpy()

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """
        Crops the input tensor along the second dimension (axis=1) to match the original size.

        Args:
            tensor (np.ndarray): A 3D NumPy array of shape (X, Y, Z).
            original_size (int): The desired size for the second dimension (Y).

        Returns:
            np.ndarray: A cropped 3D tensor with shape (X, original_size, Z).
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (X, Y, Z), got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        if current_size <= original_size:
            return tensor

        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor):
        """
        Pads the input tensor along the feature dimension to match the expected input shape.

        Args:
            input_tensor: Tensor of shape (batch_size, seq_len, channels).

        Returns:
            Padded tensor matching the model's expected input shape.
        """
        if self._framework == 'tensorflow':
            return self._padding_input_tensor_tensorflow(input_tensor)
        else:
            return self._padding_input_tensor_pytorch(input_tensor)

    def _padding_input_tensor_tensorflow(self, input_tensor):
        """TensorFlow implementation of _padding_input_tensor."""
        input_tensor = self.tf.cast(input_tensor, self.tf.float32)
        input_shape_dynamic = self.tf.shape(input_tensor)
        input_rank = self.tf.rank(input_tensor)

        target_dimension = self._network.input_shape[0][-2]
        static_channels = input_tensor.shape[-1]
        current_dimension = input_shape_dynamic[-2]
        padding_needed = self.tf.maximum(0, target_dimension - current_dimension)

        tensor_paddings = self.tf.concat([
            self.tf.zeros([input_rank - 2, 2], dtype=self.tf.int32),
            [[0, padding_needed]],
            self.tf.zeros([1, 2], dtype=self.tf.int32)
        ], axis=0)

        padded_tensor = self.tf.cond(
            self.tf.equal(padding_needed, 0),
            lambda: input_tensor,
            lambda: self.tf.pad(input_tensor, paddings=tensor_paddings, mode="CONSTANT", constant_values=0)
        )

        padded_tensor = self.tf.ensure_shape(padded_tensor, [None, target_dimension, static_channels])
        return padded_tensor

    def _padding_input_tensor_pytorch(self, input_tensor):
        """PyTorch implementation of _padding_input_tensor."""
        input_tensor = input_tensor.float()
        
        # Determine target dimension from network input
        target_dimension = self._network.input_shape[-2] if hasattr(self._network, 'input_shape') else input_tensor.shape[-2]
        current_dimension = input_tensor.shape[-2]
        padding_needed = max(0, target_dimension - current_dimension)

        if padding_needed == 0:
            return input_tensor

        # PyTorch padding format: (left, right, top, bottom, front, back)
        # We want to pad the second-to-last dimension
        pad = (0, 0, 0, padding_needed)  # (channels, feature_dim)
        padded_tensor = self.nn.functional.pad(input_tensor, pad, mode='constant', value=0)
        
        return padded_tensor

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class (dict): Dictionary specifying number of samples per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict: Dictionary where keys are class labels and values are generated samples.
        """
        if self._framework == 'tensorflow':
            return self._get_samples_tensorflow(number_samples_per_class)
        else:
            return self._get_samples_pytorch(number_samples_per_class)

    def _get_samples_tensorflow(self, number_samples_per_class):
        """TensorFlow implementation of get_samples."""
        from tensorflow.keras.utils import to_categorical
        
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            generated_samples = self.generate_data(
                numpy.array(label_samples_generated, dtype=numpy.float32),
                batch_size=64
            )

            generated_samples = numpy.rint(generated_samples)
            generated_data[label_class] = generated_samples

        return generated_data

    def _get_samples_pytorch(self, number_samples_per_class):
        """PyTorch implementation of get_samples."""
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            label_samples_generated = self.torch.zeros(
                number_instances,
                number_samples_per_class["number_classes"],
                device=self._device
            )
            label_samples_generated[:, label_class] = 1

            generated_samples = self.generate_data(
                label_samples_generated,
                batch_size=64
            )

            generated_samples = numpy.rint(generated_samples)
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save all models (encoder, decoder, and both UNets).

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if self._framework == 'tensorflow':
            self._save_model_tensorflow(directory, file_name)
        else:
            self._save_model_pytorch(directory, file_name)

    def _save_model_tensorflow(self, directory, file_name):
        """TensorFlow implementation of save_model."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        encoder_file_name = os.path.join(directory, f"{file_name}_encoder")
        decoder_file_name = os.path.join(directory, f"{file_name}_decoder")
        first_unet_file_name = os.path.join(directory, f"{file_name}_first_unet")
        second_unet_file_name = os.path.join(directory, f"{file_name}_second_unet")

        self._save_model_to_json(self._encoder_model_data, f"{encoder_file_name}.json")
        self._encoder_model_data.save_weights(f"{encoder_file_name}.weights.h5")

        self._save_model_to_json(self._decoder_model_data, f"{decoder_file_name}.json")
        self._decoder_model_data.save_weights(f"{decoder_file_name}.weights.h5")

        self._save_model_to_json(self._network, f"{first_unet_file_name}.json")
        self._network.save_weights(f"{first_unet_file_name}.weights.h5")

        self._save_model_to_json(self._second_unet_model, f"{second_unet_file_name}.json")
        self._second_unet_model.save_weights(f"{second_unet_file_name}.weights.h5")

    def _save_model_pytorch(self, directory, file_name):
        """PyTorch implementation of save_model."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        encoder_file = os.path.join(directory, f"{file_name}_encoder.pt")
        decoder_file = os.path.join(directory, f"{file_name}_decoder.pt")
        first_unet_file = os.path.join(directory, f"{file_name}_first_unet.pt")
        second_unet_file = os.path.join(directory, f"{file_name}_second_unet.pt")

        self.torch.save(self._encoder_model_data.state_dict(), encoder_file)
        self.torch.save(self._decoder_model_data.state_dict(), decoder_file)
        self.torch.save(self._network.state_dict(), first_unet_file)
        self.torch.save(self._second_unet_model.state_dict(), second_unet_file)

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file (TensorFlow only).

        Args:
            model: Model to save.
            file_path (str): Path to the JSON file.
        """
        try:
            with open(file_path, "w") as json_file:
                json.dump(model.to_json(), json_file)
            print(f"Model successfully saved to {file_path}.")
        except Exception as e:
            error_message = f"Error occurred while saving model: {str(e)}"
            with open(file_path, "w") as error_file:
                error_file.write(error_message)
            print(f"An error occurred and was saved to {file_path}: {error_message}")

    @property
    def framework(self) -> str:
        """Get the framework being used."""
        return self._framework

    @property
    def ema(self) -> Any:
        """Get the Exponential Moving Average (EMA) decay rate."""
        return self._ema

    @ema.setter
    def ema(self, value: Any) -> None:
        """Set the Exponential Moving Average (EMA) decay rate."""
        if not isinstance(value, (int, float)) or not (0 < value < 1):
            raise ValueError("EMA must be a float between 0 and 1")
        self._ema = value

    @property
    def margin(self) -> float:
        """Get the margin value used in contrastive loss."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set the margin value for contrastive loss."""
        if value <= 0:
            raise ValueError("Margin must be positive")
        self._margin = value

    @property
    def gdf_util(self) -> Any:
        """Get the Gaussian Diffusion utility."""
        return self._gdf_util

    @gdf_util.setter
    def gdf_util(self, value: Any) -> None:
        """Set the Gaussian Diffusion utility."""
        self._gdf_util = value

    @property
    def time_steps(self) -> int:
        """Get the number of diffusion time steps."""
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int) -> None:
        """Set the number of diffusion time steps."""
        if value <= 0:
            raise ValueError("Time steps must be positive")
        self._time_steps = value

    @property
    def train_stage(self) -> str:
        """Get the current training stage."""
        return self._train_stage

    @train_stage.setter
    def train_stage(self, value: str) -> None:
        """Set the current training stage."""
        self._train_stage = value

    @property
    def network(self) -> Any:
        """Get the primary U-Net model."""
        return self._network

    @network.setter
    def network(self, value: Any) -> None:
        """Set the primary U-Net model."""
        self._network = value
        if self._framework == 'pytorch':
            self._network.to(self._device)

    @property
    def second_unet_model(self) -> Any:
        """Get the secondary U-Net model."""
        return self._second_unet_model

    @second_unet_model.setter
    def second_unet_model(self, value: Any) -> None:
        """Set the secondary U-Net model."""
        self._second_unet_model = value
        if self._framework == 'pytorch':
            self._second_unet_model.to(self._device)

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension size."""
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value: int) -> None:
        """Set the embedding dimension size."""
        if value <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._embedding_dimension = value

    @property
    def encoder_model_data(self) -> Any:
        """Get the encoder model."""
        return self._encoder_model_data

    @encoder_model_data.setter
    def encoder_model_data(self, value: Any) -> None:
        """Set the encoder model."""
        self._encoder_model_data = value
        if self._framework == 'pytorch':
            self._encoder_model_data.to(self._device)

    @property
    def decoder_model_data(self) -> Any:
        """Get the decoder model."""
        return self._decoder_model_data

    @decoder_model_data.setter
    def decoder_model_data(self, value: Any) -> None:
        """Set the decoder model."""
        self._decoder_model_data = value
