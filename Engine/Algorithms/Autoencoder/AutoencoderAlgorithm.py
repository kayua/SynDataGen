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
    import logging

except ImportError as error:
    print(error)
    sys.exit(-1)


# Framework detection based on environment variable
def detect_framework():
    """
    Detects and validates the ML framework to use.

    Priority:
    1. ML_FRAMEWORK environment variable (if set)
    2. Auto-detection (TensorFlow first, then PyTorch)

    Returns:
        str: The framework name ('tensorflow' or 'pytorch')

    Raises:
        SystemExit: If no framework is available or invalid framework specified
    """
    framework_env = os.environ.get('ML_FRAMEWORK', '').lower()

    if framework_env:
        # User specified a framework via environment variable
        if framework_env == 'tensorflow':
            try:
                import tensorflow as tf
                logging.info(f"Using framework from ML_FRAMEWORK environment variable: tensorflow")
                return 'tensorflow'
            except ImportError:
                logging.error(f"ML_FRAMEWORK set to 'tensorflow' but TensorFlow is not installed.")
                sys.exit(-1)
        elif framework_env == 'pytorch':
            try:
                import torch
                logging.info(f"Using framework from ML_FRAMEWORK environment variable: pytorch")
                return 'pytorch'
            except ImportError:
                logging.error(f"ML_FRAMEWORK set to 'pytorch' but PyTorch is not installed.")
                sys.exit(-1)
        else:
            logging.error(f"Invalid ML_FRAMEWORK value '{framework_env}'. Valid options: 'tensorflow' or 'pytorch'.")
            sys.exit(-1)
    else:
        # Auto-detect available framework
        framework = None
        try:
            import tensorflow as tf
            framework = 'tensorflow'
        except ImportError:
            pass

        if framework is None:
            try:
                import torch
                framework = 'pytorch'
            except ImportError:
                pass

        if framework is None:
            logging.error("Neither TensorFlow nor PyTorch is installed. Please install one of them.")
            sys.exit(-1)
        else:
            logging.info(f"Auto-detected framework: {framework}")
            return framework


# Detect framework at module load time
DETECTED_FRAMEWORK = detect_framework()


class AutoencoderAlgorithm:
    """
    An abstract class for AutoEncoder models supporting both TensorFlow and PyTorch.

    This class provides a foundation for AutoEncoder models with methods for training,
    generating synthetic data, saving and loading models.

    Args:
        @framework (str or None):
            Framework to use: 'tensorflow', 'pytorch', or None.
            If None, uses the framework detected from ML_FRAMEWORK environment variable or auto-detection.
        @encoder_model (Model):
            The encoder part of the AutoEncoder.
        @decoder_model (Model):
            The decoder part of the AutoEncoder.
        @loss_function (Loss):
            The loss function for training.
        @file_name_encoder (str):
            The file name for saving the encoder model.
        @file_name_decoder (str):
            The file name for saving the decoder model.
        @models_saved_path (str):
            The path to save the models.
        @latent_mean_distribution (float):
            Mean of the latent space distribution.
        @latent_stander_deviation (float):
            Standard deviation of the latent space distribution.
        @latent_dimension (int):
            The dimensionality of the latent space.

    Attributes:
        @_framework (str):
            Framework being used ('tensorflow' or 'pytorch').
        @_encoder (Model):
            The encoder part of the AutoEncoder.
        @_decoder (Model):
            The decoder part of the AutoEncoder.
        @_loss_function (Loss):
            Loss function for training.
        @_total_loss_tracker:
            Metric for tracking total loss.
        @_file_name_encoder (str):
            File name for saving the encoder model.
        @_file_name_decoder (str):
            File name for saving the decoder model.
        @_models_saved_path (str):
            Path to save the models.
        @_encoder_decoder_model:
            Combined encoder-decoder model.

    Example:
        >>> # Using environment variable
        >>> # export ML_FRAMEWORK=tensorflow
        >>> autoencoder = AutoencoderAlgorithm(
        ...     framework=None,  # Will use ML_FRAMEWORK or auto-detect
        ...     encoder_model=encoder_model,
        ...     decoder_model=decoder_model,
        ...     loss_function=tensorflow.keras.losses.MeanSquaredError(),
        ...     file_name_encoder="encoder_model.h5",
        ...     file_name_decoder="decoder_model.h5",
        ...     models_saved_path="./autoencoder_models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     latent_dimension=64
        ... )
        >>> # Or explicitly specify framework (overrides environment variable)
        >>> autoencoder = AutoencoderAlgorithm(
        ...     framework='pytorch',
        ...     encoder_model=encoder_model,
        ...     decoder_model=decoder_model,
        ...     loss_function=nn.MSELoss(),
        ...     file_name_encoder="encoder_model.pt",
        ...     file_name_decoder="decoder_model.pt",
        ...     models_saved_path="./autoencoder_models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     latent_dimension=64
        ... )
    """

    def __init__(self,
                 framework=None,
                 encoder_model=None,
                 decoder_model=None,
                 loss_function=None,
                 file_name_encoder="encoder_model",
                 file_name_decoder="decoder_model",
                 models_saved_path="./autoencoder_models/",
                 latent_mean_distribution=0.0,
                 latent_stander_deviation=1.0,
                 latent_dimension=64):
        """
        Initializes an AutoEncoder model with an encoder, decoder, and necessary configurations.

        Args:
            @framework (str or None):
                Framework to use: 'tensorflow', 'pytorch', or None.
                If None, uses the framework detected from ML_FRAMEWORK environment variable or auto-detection.
            @encoder_model (Model):
                The encoder part of the AutoEncoder.
            @decoder_model (Model):
                The decoder part of the AutoEncoder.
            @loss_function (Loss):
                The loss function used for training.
            @file_name_encoder (str):
                The filename for saving the trained encoder model.
            @file_name_decoder (str):
                The filename for saving the trained decoder model.
            @models_saved_path (str):
                The directory path where models should be saved.
            @latent_mean_distribution (float):
                The mean of the latent noise distribution.
            @latent_stander_deviation (float):
                The standard deviation of the latent noise distribution.
            @latent_dimension (int):
                The number of dimensions in the latent space.
        """

        # Use detected framework if none specified
        if framework is None:
            framework = DETECTED_FRAMEWORK
            logging.info(f"No framework specified, using detected framework: {framework}")

        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        if not isinstance(file_name_encoder, str) or not file_name_encoder:
            raise ValueError("file_name_encoder must be a non-empty string.")

        if not isinstance(file_name_decoder, str) or not file_name_decoder:
            raise ValueError("file_name_decoder must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("models_saved_path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("latent_mean_distribution must be a number.")

        if not isinstance(latent_stander_deviation, (int, float)):
            raise TypeError("latent_stander_deviation must be a number.")

        if latent_stander_deviation <= 0:
            raise ValueError("latent_stander_deviation must be greater than 0.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        self._framework = framework
        self._encoder = encoder_model
        self._decoder = decoder_model
        self._loss_function = loss_function
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._latent_dimension = latent_dimension
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder
        self._models_saved_path = models_saved_path
        self.optimizer = None

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            from tensorflow.keras.metrics import Mean
            from tensorflow.keras.models import Model

            self.tf = tf
            self._total_loss_tracker = Mean(name="loss")

            if self._encoder is not None and self._decoder is not None:
                self._encoder_decoder_model = Model(self._encoder.input, self._decoder(self._encoder.output))
            else:
                self._encoder_decoder_model = None

            logging.info("Initialized AutoencoderAlgorithm with TensorFlow backend")

        else:  # pytorch
            import torch
            import torch.nn as nn

            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if self._encoder is not None:
                self._encoder.to(self._device)
            if self._decoder is not None:
                self._decoder.to(self._device)

            self._total_loss = 0.0
            self._num_batches = 0

            # Create encoder-decoder model if both models are provided
            if self._encoder is not None and self._decoder is not None:
                class EncoderDecoder(nn.Module):
                    def __init__(self, encoder, decoder):
                        super().__init__()
                        self.encoder = encoder
                        self.decoder = decoder

                    def forward(self, x):
                        encoded = self.encoder(x)
                        decoded = self.decoder(encoded)
                        return decoded

                self._encoder_decoder_model = EncoderDecoder(self._encoder, self._decoder).to(self._device)
            else:
                self._encoder_decoder_model = None

            logging.info(f"Initialized AutoencoderAlgorithm with PyTorch backend on device: {self._device}")

    def compile(self, optimizer, *args, **kwargs):
        """
        Compiles the AutoEncoder model with an optimizer.

        Args:
            optimizer: Optimizer for training.
            *args, **kwargs: Additional arguments.
        """
        self.optimizer = optimizer

    def train_step(self, batch):
        """
        Perform a training step for the AutoEncoder.

        Args:
            batch: Input data batch.

        Returns:
            dict: Dictionary containing the loss value.
        """
        if self._framework == 'tensorflow':
            return self._train_step_tensorflow(batch)
        else:
            return self._train_step_pytorch(batch)

    def _train_step_tensorflow(self, batch):
        """TensorFlow implementation of train_step."""
        batch_x, batch_y = batch

        with self.tf.GradientTape() as gradient_ae:
            reconstructed_data = self._encoder_decoder_model(batch_x, training=True)
            update_gradient_loss = self.tf.reduce_mean(self.tf.square(batch_y - reconstructed_data))

        gradient_update = gradient_ae.gradient(update_gradient_loss, self._encoder_decoder_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_update, self._encoder_decoder_model.trainable_variables))
        self._total_loss_tracker.update_state(update_gradient_loss)

        return {"loss": self._total_loss_tracker.result()}

    def _train_step_pytorch(self, batch):
        """PyTorch implementation of train_step."""
        batch_x, batch_y = batch
        batch_x = batch_x.to(self._device)
        batch_y = batch_y.to(self._device)

        self.optimizer.zero_grad()
        reconstructed_data = self._encoder_decoder_model(batch_x)
        update_gradient_loss = self.torch.mean((batch_y - reconstructed_data) ** 2)

        update_gradient_loss.backward()
        self.optimizer.step()

        self._total_loss += update_gradient_loss.item()
        self._num_batches += 1

        return {"loss": self._total_loss / self._num_batches}

    def fit(self, train_dataset, epochs=1, verbose=1):
        """
        Trains the AutoEncoder model.

        Args:
            train_dataset: Training dataset.
            epochs (int): Number of epochs.
            verbose (int): Verbosity mode.
        """
        for epoch in range(epochs):
            if self._framework == 'pytorch':
                self._total_loss = 0.0
                self._num_batches = 0

            epoch_losses = []

            for batch in train_dataset:
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict['loss'])

            if verbose:
                avg_loss = numpy.mean([float(l) for l in epoch_losses])
                logging.info(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each specified class using the trained decoder.

        Args:
            number_samples_per_class (dict):
                A dictionary specifying how many synthetic samples should be generated per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples.
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

            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,
                self._latent_stander_deviation,
                (number_instances, self._latent_dimension)
            )

            generated_samples = self._decoder.predict([latent_noise, label_samples_generated], verbose=0)
            generated_samples = numpy.rint(generated_samples)
            generated_data[label_class] = generated_samples

        return generated_data

    def _get_samples_pytorch(self, number_samples_per_class):
        """PyTorch implementation of get_samples."""
        generated_data = {}
        self._decoder.eval()

        with self.torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                label_samples_generated = self.torch.zeros(
                    number_instances,
                    number_samples_per_class["number_classes"],
                    device=self._device
                )
                label_samples_generated[:, label_class] = 1

                latent_noise = self.torch.randn(
                    number_instances,
                    self._latent_dimension,
                    device=self._device
                ) * self._latent_stander_deviation + self._latent_mean_distribution

                generated_samples = self._decoder(latent_noise, label_samples_generated)
                generated_samples = self.torch.round(generated_samples)
                generated_data[label_class] = generated_samples.cpu().numpy()

        self._decoder.train()
        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models.

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

        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder")

        self._save_model_to_json(self._encoder, f"{encoder_file_name}.json")
        self._encoder.save_weights(f"{encoder_file_name}.weights.h5")

        self._save_model_to_json(self._decoder, f"{decoder_file_name}.json")
        self._decoder.save_weights(f"{decoder_file_name}.weights.h5")

        logging.info(f"Models saved to {directory}")

    def _save_model_pytorch(self, directory, file_name):
        """PyTorch implementation of save_model."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pt")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pt")

        self.torch.save({
            'model_state_dict': self._encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, encoder_file_name)

        self.torch.save({
            'model_state_dict': self._decoder.state_dict()
        }, decoder_file_name)

        logging.info(f"Models saved to {directory}")

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file (TensorFlow only).

        Args:
            model (Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(model.to_json(), json_file)

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        if self._framework == 'tensorflow':
            self._load_models_tensorflow(directory, file_name)
        else:
            self._load_models_pytorch(directory, file_name)

    def _load_models_tensorflow(self, directory, file_name):
        """TensorFlow implementation of load_models."""
        from tensorflow.keras.models import model_from_json

        encoder_file_name = f"{file_name}_encoder"
        decoder_file_name = f"{file_name}_decoder"

        encoder_path = os.path.join(directory, f"fold_{encoder_file_name}")
        decoder_path = os.path.join(directory, f"fold_{decoder_file_name}")

        with open(f"{encoder_path}.json", 'r') as json_file:
            encoder_json = json.load(json_file)
            self._encoder = model_from_json(encoder_json)
            self._encoder.load_weights(f"{encoder_path}.weights.h5")

        with open(f"{decoder_path}.json", 'r') as json_file:
            decoder_json = json.load(json_file)
            self._decoder = model_from_json(decoder_json)
            self._decoder.load_weights(f"{decoder_path}.weights.h5")

        logging.info(f"Models loaded from {directory}")

    def _load_models_pytorch(self, directory, file_name):
        """PyTorch implementation of load_models."""
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pt")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pt")

        encoder_checkpoint = self.torch.load(encoder_file_name, map_location=self._device)
        self._encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        if self.optimizer and encoder_checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(encoder_checkpoint['optimizer_state_dict'])

        decoder_checkpoint = self.torch.load(decoder_file_name, map_location=self._device)
        self._decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

        logging.info(f"Models loaded from {directory}")

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
        if self._framework == 'pytorch':
            self._decoder.to(self._device)

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder
        if self._framework == 'pytorch':
            self._encoder.to(self._device)