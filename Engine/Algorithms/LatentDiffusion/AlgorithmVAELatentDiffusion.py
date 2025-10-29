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

except ImportError as error:
    print(error)
    sys.exit(-1)


class VAELatentDiffusionAlgorithm:
    """
    A framework-agnostic Variational AutoEncoder (VAE) model for generating synthetic data.
    Supports both TensorFlow and PyTorch.

    The model includes an encoder and a decoder for encoding input data and reconstructing
    it from a learned latent space. During training, it computes both the reconstruction loss
    and the KL divergence loss. The trained decoder can be used to generate synthetic data.

    This class supports customizable latent space parameters and loss functions, making it
    adaptable for different generative tasks.

    Attributes:
        @framework (str):
            Framework to use: 'tensorflow' or 'pytorch'.
        @_encoder (Model):
            Encoder model that encodes input data into the latent space.
        @_decoder (Model):
            Decoder model that reconstructs data from the latent representation.
        @_loss_function (callable):
            Function used to compute the total loss during training.
        @_total_loss_tracker:
            Tracks the overall loss during training.
        @_reconstruction_loss_tracker:
            Tracks the reconstruction loss during training.
        @_kl_loss_tracker:
            Tracks the KL divergence loss during training.
        @_latent_mean_distribution (float):
            Mean of the latent distribution.
        @_latent_stander_deviation (float):
            Standard deviation of the latent distribution.
        @_latent_dimension (int):
            Dimensionality of the latent space.
        @_decoder_latent_dimension (int):
            Dimensionality of the latent space used by the decoder.
        @_file_name_encoder (str):
            File name for saving the encoder model.
        @_file_name_decoder (str):
            File name for saving the decoder model.
        @_models_saved_path (str):
            Directory path where the encoder and decoder models are saved.

    Example:
        >>> # TensorFlow example
        >>> vae_model = VAELatentDiffusionAlgorithm(
        ...     framework='tensorflow',
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     loss_function=custom_loss_function,
        ...     latent_dimension=128,
        ...     decoder_latent_dimension=128,
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     file_name_encoder="encoder_model.h5",
        ...     file_name_decoder="decoder_model.h5",
        ...     models_saved_path="models/"
        ... )
        
        >>> # PyTorch example
        >>> vae_model = VAELatentDiffusionAlgorithm(
        ...     framework='pytorch',
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     loss_function=custom_loss_function,
        ...     latent_dimension=128,
        ...     decoder_latent_dimension=128,
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     file_name_encoder="encoder_model.pt",
        ...     file_name_decoder="decoder_model.pt",
        ...     models_saved_path="models/"
        ... )
    """

    def __init__(self,
                 framework,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path):
        """
        Initializes the VAELatentDiffusionAlgorithm model with provided encoder and decoder models, 
        loss function, and latent space parameters.

        Args:
            @framework (str):
                Framework to use: 'tensorflow' or 'pytorch'.
            @encoder_model (Model):
                The encoder model responsible for encoding input data into latent variables.
            @decoder_model (Model):
                The decoder model responsible for reconstructing data from the latent space.
            @loss_function (callable):
                The loss function used to compute the training loss.
            @latent_dimension (int):
                The dimensionality of the latent space.
            @decoder_latent_dimension (int):
                The dimensionality of the latent space used by the decoder.
            @latent_mean_distribution (float):
                The mean of the latent distribution (usually 0).
            @latent_stander_deviation (float):
                The standard deviation of the latent distribution (usually 1).
            @file_name_encoder (str):
                The filename for saving the encoder model.
            @file_name_decoder (str):
                The filename for saving the decoder model.
            @models_saved_path (str):
                The directory where the models will be saved.

        Raises:
            ValueError:
                If framework is not 'tensorflow' or 'pytorch'.
                If latent_dimension <= 0.
                If latent_stander_deviation <= 0.
                If file paths are invalid.
        """
        
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(decoder_latent_dimension, int) or decoder_latent_dimension <= 0:
            raise ValueError("decoder_latent_dimension must be a positive integer.")

        if not isinstance(latent_stander_deviation, (int, float)) or latent_stander_deviation <= 0:
            raise ValueError("latent_stander_deviation must be a positive number.")

        if not isinstance(file_name_encoder, str) or not file_name_encoder:
            raise ValueError("file_name_encoder must be a non-empty string.")

        if not isinstance(file_name_decoder, str) or not file_name_decoder:
            raise ValueError("file_name_decoder must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("models_saved_path must be a non-empty string.")

        self._framework = framework
        self._encoder = encoder_model
        self._decoder = decoder_model
        self._loss_function = loss_function
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder
        self._models_saved_path = models_saved_path
        self.optimizer = None

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            from tensorflow.keras.metrics import Mean
            
            self.tf = tf
            self._total_loss_tracker = Mean(name="loss")
            self._reconstruction_loss_tracker = Mean(name="reconstruction_loss")
            self._kl_loss_tracker = Mean(name="kl_loss")
            
        else:  # pytorch
            import torch
            import torch.nn as nn
            
            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._encoder.to(self._device)
            self._decoder.to(self._device)
            
            # PyTorch metrics tracking
            self._total_loss = 0.0
            self._reconstruction_loss = 0.0
            self._kl_loss = 0.0
            self._num_batches = 0

    def compile(self, optimizer, *args, **kwargs):
        """
        Compiles the VAE model with an optimizer.

        Args:
            optimizer: Optimizer for training.
            *args, **kwargs: Additional arguments.
        """
        self.optimizer = optimizer

    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).

        Args:
            batch: Input data batch (batch_x, batch_y).

        Returns:
            dict: Dictionary containing the loss values (total loss, reconstruction loss, KL divergence loss).
        """
        if self._framework == 'tensorflow':
            return self._train_step_tensorflow(batch)
        else:
            return self._train_step_pytorch(batch)

    def _train_step_tensorflow(self, batch):
        """TensorFlow implementation of train_step."""
        batch_x, batch_y = batch

        with self.tf.GradientTape() as tape:
            # Forward pass
            latent_mean, latent_log_variation, latent, label = self._encoder(batch_x)
            reconstruction_data = self._decoder([latent, label])

            # Reconstruction loss
            binary_cross_entropy_loss = self.tf.keras.losses.binary_crossentropy(batch_y, reconstruction_data)
            reconstruction_loss = self.tf.reduce_mean(binary_cross_entropy_loss)

            # KL divergence loss
            encoder_output = (1 + latent_log_variation - self.tf.square(latent_mean))
            kl_divergence_loss = -0.5 * (encoder_output - self.tf.exp(latent_log_variation))
            kl_divergence_loss = self.tf.reduce_mean(self.tf.reduce_sum(kl_divergence_loss, axis=1))

            # Total loss
            loss_model_in_reconstruction = reconstruction_loss + kl_divergence_loss

        # Compute gradients and update
        gradient_update = tape.gradient(loss_model_in_reconstruction, 
                                       list(self._encoder.trainable_variables) + 
                                       list(self._decoder.trainable_variables))
        
        trainable_vars = list(self._encoder.trainable_variables) + list(self._decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_update, trainable_vars))
        
        # Update metrics
        self._total_loss_tracker.update_state(loss_model_in_reconstruction)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._kl_loss_tracker.update_state(kl_divergence_loss)

        return {
            "loss": self._total_loss_tracker.result(),
            "reconstruction_loss": self._reconstruction_loss_tracker.result(),
            "kl_loss": self._kl_loss_tracker.result()
        }

    def _train_step_pytorch(self, batch):
        """PyTorch implementation of train_step."""
        batch_x, batch_y = batch
        batch_x = batch_x.to(self._device)
        batch_y = batch_y.to(self._device)

        self.optimizer.zero_grad()

        # Forward pass
        latent_mean, latent_log_variation, latent, label = self._encoder(batch_x)
        reconstruction_data = self._decoder(latent, label)

        # Reconstruction loss (binary cross entropy)
        reconstruction_loss = self.nn.functional.binary_cross_entropy(
            reconstruction_data, batch_y, reduction='mean'
        )

        # KL divergence loss
        kl_divergence_loss = -0.5 * self.torch.sum(
            1 + latent_log_variation - latent_mean.pow(2) - latent_log_variation.exp(),
            dim=1
        )
        kl_divergence_loss = self.torch.mean(kl_divergence_loss)

        # Total loss
        total_loss = reconstruction_loss + kl_divergence_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Update metrics
        self._total_loss += total_loss.item()
        self._reconstruction_loss += reconstruction_loss.item()
        self._kl_loss += kl_divergence_loss.item()
        self._num_batches += 1

        return {
            "loss": self._total_loss / self._num_batches,
            "reconstruction_loss": self._reconstruction_loss / self._num_batches,
            "kl_loss": self._kl_loss / self._num_batches
        }

    def fit(self, train_dataset, epochs=1, verbose=1):
        """
        Trains the VAE model.

        Args:
            train_dataset: Training dataset.
            epochs (int): Number of epochs.
            verbose (int): Verbosity mode.
        """
        for epoch in range(epochs):
            if self._framework == 'pytorch':
                self._total_loss = 0.0
                self._reconstruction_loss = 0.0
                self._kl_loss = 0.0
                self._num_batches = 0
            
            epoch_losses = []
            
            for batch in train_dataset:
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict)
            
            if verbose:
                avg_loss = numpy.mean([l['loss'] for l in epoch_losses])
                avg_recon = numpy.mean([l['reconstruction_loss'] for l in epoch_losses])
                avg_kl = numpy.mean([l['kl_loss'] for l in epoch_losses])
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
                      f"reconstruction_loss: {avg_recon:.4f} - kl_loss: {avg_kl:.4f}")

    def get_decoder_trained(self):
        """Returns the trained decoder model."""
        return self._decoder

    def get_encoder_trained(self):
        """Returns the trained encoder model."""
        return self._encoder

    def create_embedding(self, data):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data: Input data to encode.

        Returns:
            ndarray: Latent space representations.
        """
        if self._framework == 'tensorflow':
            return self._encoder.predict(data, batch_size=32)[0]
        else:
            self._encoder.eval()
            with self.torch.no_grad():
                if isinstance(data, numpy.ndarray):
                    data = self.torch.tensor(data, dtype=self.torch.float32, device=self._device)
                latent_mean, _, _, _ = self._encoder(data)
            self._encoder.train()
            return latent_mean.cpu().numpy()

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.

        Args:
            number_samples_per_class (dict):
                Dictionary specifying the number of samples to generate for each class.
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
                size=(number_instances, self._decoder_latent_dimension)
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
                    self._decoder_latent_dimension,
                    device=self._device
                ) * self._latent_stander_deviation + self._latent_mean_distribution

                generated_samples = self._decoder(latent_noise, label_samples_generated)
                generated_samples = self.torch.round(generated_samples)
                generated_data[label_class] = generated_samples.cpu().numpy()

        self._decoder.train()
        return generated_data

    def generate_synthetic_data(self, number_samples_generate, labels, latent_dimension):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).

        Args:
            number_samples_generate (int): Number of synthetic samples to generate.
            labels: Labels for the generated data.
            latent_dimension (int): Dimension of the latent space.

        Returns:
            ndarray: Synthetic data generated by the decoder.
        """
        if self._framework == 'tensorflow':
            return self._generate_synthetic_data_tensorflow(number_samples_generate, labels, latent_dimension)
        else:
            return self._generate_synthetic_data_pytorch(number_samples_generate, labels, latent_dimension)

    def _generate_synthetic_data_tensorflow(self, number_samples_generate, labels, latent_dimension):
        """TensorFlow implementation of generate_synthetic_data."""
        random_noise_generate = self.tf.random.normal(
            shape=(number_samples_generate, latent_dimension),
            mean=self._latent_mean_distribution,
            stddev=self._latent_stander_deviation,
            dtype=self.tf.float32
        )

        label_list = self.tf.cast(
            self.tf.fill((number_samples_generate, 1), labels),
            dtype=self.tf.float32
        )

        synthetic_data = self._decoder.predict([random_noise_generate.numpy(), label_list.numpy()])
        return synthetic_data

    def _generate_synthetic_data_pytorch(self, number_samples_generate, labels, latent_dimension):
        """PyTorch implementation of generate_synthetic_data."""
        self._decoder.eval()
        
        with self.torch.no_grad():
            random_noise_generate = self.torch.randn(
                number_samples_generate,
                latent_dimension,
                device=self._device
            ) * self._latent_stander_deviation + self._latent_mean_distribution

            label_list = self.torch.full(
                (number_samples_generate, 1),
                labels,
                dtype=self.torch.float32,
                device=self._device
            )

            synthetic_data = self._decoder(random_noise_generate, label_list)
        
        self._decoder.train()
        return synthetic_data.cpu().numpy()

    @property
    def metrics(self):
        """Returns list of metrics to track during training."""
        if self._framework == 'tensorflow':
            return [self._total_loss_tracker, self._reconstruction_loss_tracker, self._kl_loss_tracker]
        else:
            return ["loss", "reconstruction_loss", "kl_loss"]

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

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file (TensorFlow only).

        Args:
            model: Model to save.
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

    @property
    def framework(self):
        """Get the framework being used."""
        return self._framework

    @property
    def decoder(self):
        """Get the decoder model."""
        return self._decoder

    @property
    def encoder(self):
        """Get the encoder model."""
        return self._encoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder model."""
        self._decoder = decoder
        if self._framework == 'pytorch':
            self._decoder.to(self._device)

    @encoder.setter
    def encoder(self, encoder):
        """Set the encoder model."""
        self._encoder = encoder
        if self._framework == 'pytorch':
            self._encoder.to(self._device)

    @property
    def device(self):
        """Get the device being used (PyTorch only)."""
        return self._device if self._framework == 'pytorch' else None
