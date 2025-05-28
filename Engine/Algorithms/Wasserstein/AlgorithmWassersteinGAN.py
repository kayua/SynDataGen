#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/04/25'
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

try:
    import os
    import sys

    import json
    import numpy

    import tensorflow

    from abc import ABC

    from typing import Any

    from typing import Callable

    from tensorflow.keras import Model

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    print(error)
    sys.exit(-1)


class WassersteinAlgorithm(Model):
    """
    Implementation of the WassersteinGP Generative Adversarial Network (WGAN) algorithm.
    This class extends the Keras Model class to create a trainable WGAN model.

    The WassersteinGP GAN (Arjovsky et al., 2017) improves upon standard GANs by:
    - Using the WassersteinGP distance (Earth Mover's distance) as the loss metric
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
    "WassersteinGP Generative Adversarial Networks."
    Proceedings of the 34th International Conference on Machine Learning, PMLR 70:214-223.
    Available at: http://proceedings.mlr.press/v70/arjovsky17a.html

    Key Components:
    ---------------
    - Generator model that creates synthetic samples
    - Critic model (instead of discriminator) that scores sample realism
    - Weight clipping to enforce Lipschitz constraint (implemented in optimizer)
    - Custom training step with multiple critic updates per generator update

    Raises:
        ValueError:
            Raised if:
            - The latent dimension is non-positive.
            - The smoothing rate is outside the valid range (0, 1).
            - The number of discriminator steps is non-positive.

    Example:
        >>> python
        ... # Define generator and critic models
        ... generator = make_generator()  # Your generator implementation
        ... critic = make_critic()       # Your critic implementation
        ...
        ... # Initialize WGAN
        ... wgan = WassersteinAlgorithm(
        ... generator_model=generator,
        ... discriminator_model=critic,
        ... latent_dimension=128,
        ... generator_loss_fn=wasserstein_generator_loss,
        ... discriminator_loss_fn=wasserstein_critic_loss,
        ... file_name_discriminator='wgan_critic_weights.h5',
        ... file_name_generator='wgan_generator_weights.h5',
        ... models_saved_path='./saved_models/',
        ... latent_mean_distribution=0.0,
        ... latent_stander_deviation=1.0,
        ... smoothing_rate=0.1,
        ... discriminator_steps=5
        ... )
        >>> wgan.compile()

    """
    def __init__(self,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 generator_loss_fn,
                 discriminator_loss_fn,
                 file_name_discriminator,
                 file_name_generator,
                 models_saved_path,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 smoothing_rate,
                 discriminator_steps,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # Initialize instance variables with provided or default values
        self._generator_optimizer = None
        self._discriminator_optimizer = None
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_loss_fn = generator_loss_fn
        self._smooth_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator, loss_discriminator, *args, **kwargs):

        super().compile()
        self._discriminator_optimizer = optimizer_discriminator
        self._generator_optimizer = optimizer_generator
        self._discriminator_loss_fn = loss_discriminator
        self._generator_loss_fn = loss_generator

    @tensorflow.function
    def train_step(self, batch):
        """
        Execute one training step (forward pass + backward pass) for WGAN.

        The training step consists of:
        1. Multiple critic updates (default 5 as per WGAN paper)
        2. One generator update
        3. Weight clipping (should be handled by the optimizer)

        Args:
            batch (tuple): Batch of real samples (features, labels)

        Returns:
            dict: Dictionary containing loss metrics:
                - d_loss: Critic loss
                - g_loss: Generator loss
        """
        # Unpack batch into features and labels.
        real_feature, real_samples_label = batch
        batch_size = tensorflow.shape(real_feature)[0]

        # Expand label dimensions to match input expectations (e.g., (batch_size, 1)).
        real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

        # === Discriminator Training Loop ===
        for _ in range(self._discriminator_steps):
            # Generate random noise vectors for the latent space.
            latent_space = tensorflow.random.normal((batch_size, self._latent_dimension),
                                                    mean=self._latent_mean_distribution,
                                                    stddev=self._latent_stander_deviation)

            with tensorflow.GradientTape() as discriminator_gradient:
                # Generate synthetic samples from the generator using noise and labels.
                synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

                # Predict "real/fake" labels using the discriminator for real and synthetic samples.
                label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
                label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

                # Compute discriminator loss (real vs fake).
                discriminator_loss_result = self._discriminator_loss_fn(
                    real_img=label_predicted_real, fake_img=label_predicted_synthetic)

                # Combine loss.
                all_discriminator_loss = discriminator_loss_result

            # Compute and apply gradients to update the discriminator's weights.
            gradient_computed = discriminator_gradient.gradient(discriminator_loss_result,
                                                                self.discriminator.trainable_variables)
            self._discriminator_optimizer.apply_gradients(zip(gradient_computed,
                                                              self.discriminator.trainable_variables))

        # === Generator Training Step ===
        # Generate fresh random noise vectors for the latent space.
        latent_space = tensorflow.random.normal((batch_size, self._latent_dimension),
                                                mean=self._latent_mean_distribution,
                                                stddev=self._latent_stander_deviation)

        with tensorflow.GradientTape() as generator_gradient:
            # Generate synthetic samples from the generator.
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

            # Predict "real/fake" labels for synthetic samples using the discriminator.
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Compute generator loss (how well generator fools the discriminator).
            all_generator_loss = self._generator_loss_fn(predicted_labels)

        # Compute and apply gradients to update the generator's weights.
        gradient_computed = generator_gradient.gradient(all_generator_loss, self._generator.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(gradient_computed, self._generator.trainable_variables))

        # Return the loss values for monitoring/tracking.
        return {"d_loss": all_discriminator_loss, "g_loss": all_generator_loss}

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        This method creates samples conditioned on class labels, using random noise vectors
        and the generator to produce the samples.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to the number of samples to generate for each class.
                - "number_classes" (int): Total number of classes (used for one-hot encoding).

        Returns:
            dict: A dictionary where each key is a class label and the value is an array of generated samples.
        """

        # Dictionary to store generated samples for each class.
        generated_data = {}

        # Loop through each class and the desired number of samples for that class.
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for all samples of the current class.
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            # Sample random noise vectors from a normal distribution.
            latent_noise = numpy.random.normal(loc=self._latent_mean_distribution,
                                               scale=self._latent_stander_deviation,
                                               size=(number_instances, self._latent_dimension))

            # Generate synthetic samples using the generator.
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Round the generated samples to integer values
            # (if samples are intended to be binary, e.g., images with pixel values 0 or 1).
            generated_samples = numpy.rint(generated_samples)

            # Store generated samples for the current class.
            generated_data[label_class] = generated_samples

        # Return the dictionary containing generated samples for all requested classes.
        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models in both JSON and H5 formats.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        # Save encoder model
        self._save_model_to_json(self._generator, f"{generator_file_name}.json")
        self._generator.save_weights(f"{generator_file_name}.weights.h5")

        # Save decoder model
        self._save_model_to_json(self._discriminator, f"{discriminator_file_name}.json")
        self._discriminator.save_weights(f"{discriminator_file_name}.weights.h5")


    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (tf.keras.Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(model.to_json(), json_file)


    def load_models(self, directory, file_name):
        """
        Load the generator and discriminator models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for generator and discriminator models
        generator_file_name = "{}_generator".format(file_name)
        discriminator_file_name = "{}_discriminator".format(file_name)

        # Load the generator and discriminator models from the specified directory
        self._generator = self._save_neural_network_model(generator_file_name, directory)
        self._discriminator = self._save_neural_network_model(discriminator_file_name, directory)

    @property
    def discriminator(self) -> Any:
        """Get the discriminator model instance.

        Returns:
            The discriminator model used in GAN training.
        """
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value: Any) -> None:
        """Set the discriminator model instance.

        Args:
            value: The discriminator model to set.
        """
        self._discriminator = value

    @property
    def latent_dimension(self) -> int:
        """Get the dimension of the latent space.

        Returns:
            The size of the latent space dimension (positive integer).
        """
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Set the dimension of the latent space.

        Args:
            value: The latent dimension size (must be positive).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self) -> Callable:
        """Get the discriminator loss function.

        Returns:
            The loss function used for discriminator training.
        """
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value: Callable) -> None:
        """Set the discriminator loss function.

        Args:
            value: The loss function to use for discriminator training.
        """
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self) -> Callable:
        """Get the generator loss function.

        Returns:
            The loss function used for generator training.
        """
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value: Callable) -> None:
        """Set the generator loss function.

        Args:
            value: The loss function to use for generator training.
        """
        self._generator_loss_fn = value

    @property
    def gradient_penalty_weight(self) -> float:
        """Get the weight for gradient penalty in WGAN-GP.

        Returns:
            The weight factor for gradient penalty term.
        """
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value: float) -> None:
        """Set the weight for gradient penalty in WGAN-GP.

        Args:
            value: The penalty weight (must be non-negative).

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

    @property
    def smooth_rate(self) -> float:
        """Get the label smoothing rate.

        Returns:
            The rate used for one-sided label smoothing.
        """
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value: float) -> None:
        """Set the label smoothing rate.

        Args:
            value: The smoothing rate (typically between 0 and 0.3).

        Raises:
            ValueError: If value is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self) -> float:
        """Get the mean of the latent space distribution.

        Returns:
            The mean value used for latent space sampling.
        """
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value: float) -> None:
        """Set the mean of the latent space distribution.

        Args:
            value: The mean value for latent distribution.
        """
        self._latent_mean_distribution = value

    @property
    def latent_stander_deviation(self) -> float:
        """Get the standard deviation of the latent space distribution.

        Returns:
            The standard deviation used for latent space sampling.
        """
        return self._latent_stander_deviation

    @latent_stander_deviation.setter
    def latent_stander_deviation(self, value: float) -> None:
        """Set the standard deviation of the latent space distribution.

        Args:
            value: The standard deviation (must be positive).

        Raises:
            ValueError: If value is not positive.
        """
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_stander_deviation = value

    @property
    def file_name_discriminator(self) -> str:
        """Get the discriminator model save filename.

        Returns:
            The filename pattern for saving discriminator models.
        """
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_discriminator = value

    @property
    def file_name_generator(self) -> str:
        """Get the generator model save filename.

        Returns:
            The filename pattern for saving generator models.
        """
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value: str) -> None:
        """Set the generator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_generator = value

    @property
    def models_saved_path(self) -> str:
        """Get the path for saving models.

        Returns:
            The directory path where models are saved.
        """
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value: str) -> None:
        """Set the path for saving models.

        Args:
            value: The directory path to use for saving models.
        """
        self._models_saved_path = value

    @property
    def discriminator_steps(self) -> int:
        """Get the number of discriminator steps per iteration.

        Returns:
            The number of discriminator training steps per GAN iteration.
        """
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value: int) -> None:
        """Set the number of discriminator steps per iteration.

        Args:
            value: The number of steps (must be positive integer).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value