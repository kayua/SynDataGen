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


class WassersteinAlgorithm:
    """
    Implementation of the original Wasserstein Generative Adversarial Network (WGAN) algorithm.
    Supports both TensorFlow and PyTorch frameworks.

    The original WGAN (Arjovsky et al., 2017) improves upon standard GANs by:
    - Using the Wasserstein (Earth Mover's) distance as the loss metric
    - Providing more stable training dynamics
    - Offering meaningful loss metrics that correlate with generation quality

    Mathematical Formulation:
    ------------------------
    The WGAN objective function is:

        min_G max_{D ∈ 1-Lipschitz} E[D(x)] - E[D(G(z))]

    where:
        - G is the generator
        - D is the critic (discriminator)
        - x ~ P_r (real data distribution)
        - z ~ P_z (noise distribution)
        - The critic must be 1-Lipschitz continuous (enforced via weight clipping)

    Reference:
    ----------
    Arjovsky, M., Chintala, S., & Bottou, L. (2017).
    "Wasserstein Generative Adversarial Networks."
    Proceedings of the 34th International Conference on Machine Learning, PMLR 70:214-223.
    Available at: http://proceedings.mlr.press/v70/arjovsky17a.html

    Args:
        @framework (str or None):
            Framework to use: 'tensorflow', 'pytorch', or None.
            If None, uses the framework detected from ML_FRAMEWORK environment variable or auto-detection.
        @generator_model (Model):
            The generator model.
        @discriminator_model (Model):
            The discriminator/critic model.
        @latent_dimension (int):
            The dimensionality of the latent space.
        @generator_loss_fn (Callable):
            Loss function for the generator.
        @discriminator_loss_fn (Callable):
            Loss function for the discriminator.
        @file_name_discriminator (str):
            File name for saving the discriminator model.
        @file_name_generator (str):
            File name for saving the generator model.
        @models_saved_path (str):
            Directory path for saving models.
        @latent_mean_distribution (float):
            Mean of the latent space distribution.
        @latent_standard_deviation (float):
            Standard deviation of the latent space distribution.
        @smoothing_rate (float):
            Label smoothing rate.
        @discriminator_steps (int):
            Number of discriminator training steps per generator step.
        @clip_value (float):
            Value for weight clipping (default: 0.01).

    Example:
        >>> # Using environment variable
        >>> # export ML_FRAMEWORK=tensorflow
        >>> wgan = WassersteinAlgorithm(
        ...     framework=None,  # Will use ML_FRAMEWORK or auto-detect
        ...     generator_model=generator,
        ...     discriminator_model=discriminator,
        ...     latent_dimension=100,
        ...     generator_loss_fn=generator_loss,
        ...     discriminator_loss_fn=discriminator_loss,
        ...     file_name_discriminator='discriminator.h5',
        ...     file_name_generator='generator.h5',
        ...     models_saved_path='./models/',
        ...     latent_mean_distribution=0.0,
        ...     latent_standard_deviation=1.0,
        ...     smoothing_rate=0.1,
        ...     discriminator_steps=5,
        ...     clip_value=0.01
        ... )
    """

    def __init__(self,
                 framework=None,
                 generator_model=None,
                 discriminator_model=None,
                 latent_dimension=100,
                 generator_loss_fn=None,
                 discriminator_loss_fn=None,
                 file_name_discriminator="discriminator_model",
                 file_name_generator="generator_model",
                 models_saved_path="./models/",
                 latent_mean_distribution=0.0,
                 latent_standard_deviation=1.0,
                 smoothing_rate=0.1,
                 discriminator_steps=5,
                 clip_value=0.01):
        """
        Initializes the WassersteinAlgorithm with provided models and parameters.

        Args:
            @framework (str or None):
                Framework to use: 'tensorflow', 'pytorch', or None.
                If None, uses the framework detected from ML_FRAMEWORK environment variable or auto-detection.
            [Other parameters as documented in class docstring]
        """

        # Use detected framework if none specified
        if framework is None:
            framework = DETECTED_FRAMEWORK
            logging.info(f"No framework specified, using detected framework: {framework}")

        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer")

        if not isinstance(discriminator_steps, int) or discriminator_steps <= 0:
            raise ValueError("Discriminator steps must be a positive integer")

        if not 0 <= smoothing_rate <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")

        if latent_standard_deviation <= 0:
            raise ValueError("Standard deviation must be positive")

        if clip_value <= 0:
            raise ValueError("Clip value must be positive")

        self._framework = framework
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._generator_loss_fn = generator_loss_fn
        self._discriminator_loss_fn = discriminator_loss_fn
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._smooth_rate = smoothing_rate
        self._discriminator_steps = discriminator_steps
        self._clip_value = clip_value
        self._generator_optimizer = None
        self._discriminator_optimizer = None

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            self.tf = tf
            logging.info("Initialized WassersteinAlgorithm with TensorFlow backend")

        else:  # pytorch
            import torch
            import torch.nn as nn

            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if self._generator is not None:
                self._generator.to(self._device)
            if self._discriminator is not None:
                self._discriminator.to(self._device)

            self._d_loss_total = 0.0
            self._g_loss_total = 0.0
            self._num_batches = 0

            logging.info(f"Initialized WassersteinAlgorithm with PyTorch backend on device: {self._device}")

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator=None, loss_discriminator=None):
        """
        Compiles the WGAN model with optimizers and loss functions.

        Args:
            optimizer_generator: Optimizer for the generator.
            optimizer_discriminator: Optimizer for the discriminator.
            loss_generator: Loss function for the generator (optional, uses init value if None).
            loss_discriminator: Loss function for the discriminator (optional, uses init value if None).
        """
        self._generator_optimizer = optimizer_generator
        self._discriminator_optimizer = optimizer_discriminator

        if loss_generator is not None:
            self._generator_loss_fn = loss_generator
        if loss_discriminator is not None:
            self._discriminator_loss_fn = loss_discriminator

    def train_step(self, batch):
        """
        Perform a training step for the WGAN.

        Args:
            batch: Input data batch (features, labels).

        Returns:
            dict: Dictionary containing discriminator and generator losses.
        """
        if self._framework == 'tensorflow':
            return self._train_step_tensorflow(batch)
        else:
            return self._train_step_pytorch(batch)

    def _train_step_tensorflow(self, batch):
        """TensorFlow implementation of train_step."""
        real_feature, real_samples_label = batch
        batch_size = self.tf.shape(real_feature)[0]
        real_samples_label = self.tf.expand_dims(real_samples_label, axis=-1)

        # === Critic (Discriminator) Training ===
        for _ in range(self._discriminator_steps):
            latent_space = self.tf.random.normal(
                (batch_size, self._latent_dimension),
                mean=self._latent_mean_distribution,
                stddev=self._latent_standard_deviation
            )

            with self.tf.GradientTape() as disc_tape:
                fake_feature = self._generator([latent_space, real_samples_label], training=False)
                real_output = self._discriminator([real_feature, real_samples_label], training=True)
                fake_output = self._discriminator([fake_feature, real_samples_label], training=True)
                d_loss = self._discriminator_loss_fn(real_output, fake_output)

            gradients = disc_tape.gradient(d_loss, self._discriminator.trainable_variables)
            self._discriminator_optimizer.apply_gradients(zip(gradients, self._discriminator.trainable_variables))

            # Apply weight clipping to enforce 1-Lipschitz constraint
            for weight in self._discriminator.trainable_weights:
                weight.assign(self.tf.clip_by_value(weight, -self._clip_value, self._clip_value))

        # === Generator Training ===
        latent_space = self.tf.random.normal(
            (batch_size, self._latent_dimension),
            mean=self._latent_mean_distribution,
            stddev=self._latent_standard_deviation
        )

        with self.tf.GradientTape() as gen_tape:
            fake_feature = self._generator([latent_space, real_samples_label], training=True)
            fake_output = self._discriminator([fake_feature, real_samples_label], training=False)
            g_loss = self._generator_loss_fn(fake_output)

        gradients = gen_tape.gradient(g_loss, self._generator.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(gradients, self._generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def _train_step_pytorch(self, batch):
        """PyTorch implementation of train_step."""
        real_feature, real_samples_label = batch
        real_feature = real_feature.to(self._device)
        real_samples_label = real_samples_label.to(self._device)

        if real_samples_label.dim() == 1:
            real_samples_label = real_samples_label.unsqueeze(-1)

        batch_size = real_feature.shape[0]

        # === Critic (Discriminator) Training ===
        for _ in range(self._discriminator_steps):
            latent_space = self.torch.randn(
                batch_size, self._latent_dimension,
                device=self._device
            ) * self._latent_standard_deviation + self._latent_mean_distribution

            self._discriminator_optimizer.zero_grad()

            with self.torch.no_grad():
                fake_feature = self._generator(latent_space, real_samples_label)

            real_output = self._discriminator(real_feature, real_samples_label)
            fake_output = self._discriminator(fake_feature, real_samples_label)
            d_loss = self._discriminator_loss_fn(real_output, fake_output)

            d_loss.backward()
            self._discriminator_optimizer.step()

            # Apply weight clipping to enforce 1-Lipschitz constraint
            for param in self._discriminator.parameters():
                param.data.clamp_(-self._clip_value, self._clip_value)

        # === Generator Training ===
        latent_space = self.torch.randn(
            batch_size, self._latent_dimension,
            device=self._device
        ) * self._latent_standard_deviation + self._latent_mean_distribution

        self._generator_optimizer.zero_grad()

        fake_feature = self._generator(latent_space, real_samples_label)

        with self.torch.no_grad():
            self._discriminator.eval()
        fake_output = self._discriminator(fake_feature, real_samples_label)
        self._discriminator.train()

        g_loss = self._generator_loss_fn(fake_output)

        g_loss.backward()
        self._generator_optimizer.step()

        self._d_loss_total += d_loss.item()
        self._g_loss_total += g_loss.item()
        self._num_batches += 1

        return {
            "d_loss": self._d_loss_total / self._num_batches,
            "g_loss": self._g_loss_total / self._num_batches
        }

    def fit(self, train_dataset, epochs=1, verbose=1):
        """
        Trains the WGAN model.

        Args:
            train_dataset: Training dataset.
            epochs (int): Number of epochs.
            verbose (int): Verbosity mode.
        """
        for epoch in range(epochs):
            if self._framework == 'pytorch':
                self._d_loss_total = 0.0
                self._g_loss_total = 0.0
                self._num_batches = 0

            epoch_d_losses = []
            epoch_g_losses = []

            for batch in train_dataset:
                loss_dict = self.train_step(batch)
                epoch_d_losses.append(loss_dict['d_loss'])
                epoch_g_losses.append(loss_dict['g_loss'])

            if verbose:
                avg_d_loss = numpy.mean([float(l) for l in epoch_d_losses])
                avg_g_loss = numpy.mean([float(l) for l in epoch_g_losses])
                logging.info(f"Epoch {epoch + 1}/{epochs} - d_loss: {avg_d_loss:.4f} - g_loss: {avg_g_loss:.4f}")

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to the number of samples to generate.
                - "number_classes" (int): Total number of classes (for one-hot encoding).

        Returns:
            dict: A dictionary where each key is a class label and value is array of generated samples.
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
                loc=self._latent_mean_distribution,
                scale=self._latent_standard_deviation,
                size=(number_instances, self._latent_dimension)
            )

            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)
            generated_samples = numpy.rint(generated_samples)
            generated_data[label_class] = generated_samples

        return generated_data

    def _get_samples_pytorch(self, number_samples_per_class):
        """PyTorch implementation of get_samples."""
        generated_data = {}
        self._generator.eval()

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
                ) * self._latent_standard_deviation + self._latent_mean_distribution

                generated_samples = self._generator(latent_noise, label_samples_generated)
                generated_samples = self.torch.round(generated_samples)
                generated_data[label_class] = generated_samples.cpu().numpy()

        self._generator.train()
        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the generator and discriminator models.

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

        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        self._save_model_to_json(self._generator, f"{generator_file_name}.json")
        self._generator.save_weights(f"{generator_file_name}.weights.h5")

        self._save_model_to_json(self._discriminator, f"{discriminator_file_name}.json")
        self._discriminator.save_weights(f"{discriminator_file_name}.weights.h5")

        logging.info(f"Models saved to {directory}")

    def _save_model_pytorch(self, directory, file_name):
        """PyTorch implementation of save_model."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator.pt")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator.pt")

        self.torch.save({
            'model_state_dict': self._generator.state_dict(),
            'optimizer_state_dict': self._generator_optimizer.state_dict() if self._generator_optimizer else None
        }, generator_file_name)

        self.torch.save({
            'model_state_dict': self._discriminator.state_dict(),
            'optimizer_state_dict': self._discriminator_optimizer.state_dict() if self._discriminator_optimizer else None
        }, discriminator_file_name)

        logging.info(f"Models saved to {directory}")

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file (TensorFlow only).

        Args:
            model (Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        try:
            with open(file_path, "w") as json_file:
                json.dump(model.to_json(), json_file)
            logging.info(f"Model architecture saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving model to JSON: {e}")

    def load_models(self, directory, file_name):
        """
        Load the generator and discriminator models from a directory.

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

        generator_file_name = f"{file_name}_generator"
        discriminator_file_name = f"{file_name}_discriminator"

        generator_path = os.path.join(directory, f"fold_{generator_file_name}")
        discriminator_path = os.path.join(directory, f"fold_{discriminator_file_name}")

        with open(f"{generator_path}.json", 'r') as json_file:
            generator_json = json.load(json_file)
            self._generator = model_from_json(generator_json)
            self._generator.load_weights(f"{generator_path}.weights.h5")

        with open(f"{discriminator_path}.json", 'r') as json_file:
            discriminator_json = json.load(json_file)
            self._discriminator = model_from_json(discriminator_json)
            self._discriminator.load_weights(f"{discriminator_path}.weights.h5")

        logging.info(f"Models loaded from {directory}")

    def _load_models_pytorch(self, directory, file_name):
        """PyTorch implementation of load_models."""
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator.pt")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator.pt")

        generator_checkpoint = self.torch.load(generator_file_name, map_location=self._device)
        self._generator.load_state_dict(generator_checkpoint['model_state_dict'])
        if self._generator_optimizer and generator_checkpoint.get('optimizer_state_dict'):
            self._generator_optimizer.load_state_dict(generator_checkpoint['optimizer_state_dict'])

        discriminator_checkpoint = self.torch.load(discriminator_file_name, map_location=self._device)
        self._discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
        if self._discriminator_optimizer and discriminator_checkpoint.get('optimizer_state_dict'):
            self._discriminator_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])

        logging.info(f"Models loaded from {directory}")

    # Properties
    @property
    def framework(self):
        """Get the framework being used."""
        return self._framework

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value
        if self._framework == 'pytorch':
            self._discriminator.to(self._device)

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        self._generator = value
        if self._framework == 'pytorch':
            self._generator.to(self._device)

    @property
    def latent_dimension(self):
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self):
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value):
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self):
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value):
        self._generator_loss_fn = value

    @property
    def smooth_rate(self):
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value):
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self):
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value):
        self._latent_mean_distribution = value

    @property
    def latent_standard_deviation(self):
        return self._latent_standard_deviation

    @latent_standard_deviation.setter
    def latent_standard_deviation(self, value):
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_standard_deviation = value

    @property
    def file_name_discriminator(self):
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value):
        self._file_name_discriminator = value

    @property
    def file_name_generator(self):
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value):
        self._file_name_generator = value

    @property
    def models_saved_path(self):
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value):
        self._models_saved_path = value

    @property
    def discriminator_steps(self):
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value

    @property
    def clip_value(self):
        return self._clip_value

    @clip_value.setter
    def clip_value(self, value):
        if value <= 0:
            raise ValueError("Clip value must be positive")
        self._clip_value = value

    @property
    def device(self):
        """Get the device being used (PyTorch only)."""
        return self._device if self._framework == 'pytorch' else None