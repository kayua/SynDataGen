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


class WassersteinGPAlgorithm:
    """
    A framework-agnostic WassersteinGP Generative Adversarial Network (WassersteinGP GAN) model.

    This class represents a WassersteinGP GAN consisting of a generator and discriminator model.
    It implements the WassersteinGP loss with gradient penalty to improve the training of the 
    discriminator and generator. Supports both TensorFlow and PyTorch frameworks.

    Reference:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). WassersteinGP GAN.
        In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 214-223).
        http://proceedings.mlr.press/v70/arjovsky17a.html

    Args:
        @framework (str):
            Framework to use: 'tensorflow' or 'pytorch'.
        @generator_model (Model):
            The generator model responsible for generating synthetic data.
        @discriminator_model (Model):
            The discriminator model used to evaluate the authenticity of generated data.
        @latent_dimension (int):
            The dimension of the latent space from which the generator takes input.
        @generator_loss_fn (function):
            Loss function used for training the generator.
        @discriminator_loss_fn (function):
            Loss function used for training the discriminator.
        @file_name_discriminator (str):
            File name for saving/loading the discriminator model.
        @file_name_generator (str):
            File name for saving/loading the generator model.
        @models_saved_path (str):
            Path where the models are saved.
        @latent_mean_distribution (float):
            Mean of the latent space distribution.
        @latent_stander_deviation (float):
            Standard deviation of the latent space distribution.
        @smoothing_rate (float):
            Rate for label smoothing applied to the discriminator's true labels.
        @gradient_penalty_weight (float):
            Weight for the gradient penalty term in the WassersteinGP loss.
        @discriminator_steps (int):
            Number of discriminator updates per generator update.

    Raises:
        ValueError:
            Raised if:
            - The framework is not 'tensorflow' or 'pytorch'.
            - The latent dimension is non-positive.
            - The gradient penalty weight is non-positive.
            - The smoothing rate is outside the valid range (0, 1).
            - The number of discriminator steps is non-positive.

    Example:
        >>> # TensorFlow example
        >>> wgan = WassersteinGPAlgorithm(
        ...     framework='tensorflow',
        ...     generator_model=generator,
        ...     discriminator_model=discriminator,
        ...     latent_dimension=100,
        ...     generator_loss_fn=generator_loss_fn,
        ...     discriminator_loss_fn=discriminator_loss_fn,
        ...     file_name_discriminator='discriminator_model.h5',
        ...     file_name_generator='generator_model.h5',
        ...     models_saved_path='./models/',
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1,
        ...     gradient_penalty_weight=10.0,
        ...     discriminator_steps=5
        ... )
        >>> # PyTorch example
        >>> wgan = WassersteinGPAlgorithm(
        ...     framework='pytorch',
        ...     generator_model=generator,
        ...     discriminator_model=discriminator,
        ...     latent_dimension=100,
        ...     generator_loss_fn=generator_loss_fn,
        ...     discriminator_loss_fn=discriminator_loss_fn,
        ...     file_name_discriminator='discriminator_model.pt',
        ...     file_name_generator='generator_model.pt',
        ...     models_saved_path='./models/',
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1,
        ...     gradient_penalty_weight=10.0,
        ...     discriminator_steps=5
        ... )
    """

    def __init__(self,
                 framework,
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
                 gradient_penalty_weight,
                 discriminator_steps):

        # Validate framework
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        # Validate parameters
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer")

        if gradient_penalty_weight < 0:
            raise ValueError("Gradient penalty weight cannot be negative")

        if not 0 <= smoothing_rate <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")

        if not isinstance(discriminator_steps, int) or discriminator_steps <= 0:
            raise ValueError("Discriminator steps must be a positive integer")

        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("file_name_generator must be a non-empty string.")

        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("file_name_discriminator must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("models_saved_path must be a non-empty string.")

        # Initialize instance variables
        self._framework = framework
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_loss_fn = generator_loss_fn
        self._gradient_penalty_weight = gradient_penalty_weight
        self._smooth_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps
        self._generator_optimizer = None
        self._discriminator_optimizer = None

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            from tensorflow.keras.metrics import Mean
            
            self.tf = tf
            self._d_loss_tracker = Mean(name="d_loss")
            self._g_loss_tracker = Mean(name="g_loss")
            
        else:  # pytorch
            import torch
            import torch.nn as nn
            
            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._generator.to(self._device)
            self._discriminator.to(self._device)
            self._d_loss_total = 0.0
            self._g_loss_total = 0.0
            self._num_batches = 0

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator, loss_discriminator):
        """
        Compile the WassersteinGP Generative Adversarial Network (WGAN) with custom optimizers and loss functions.

        Args:
            optimizer_generator:
                The optimizer for the generator.
            optimizer_discriminator:
                The optimizer for the discriminator.
            loss_generator:
                The loss function for the generator.
            loss_discriminator:
                The loss function for the discriminator.
        """
        self._discriminator_optimizer = optimizer_discriminator
        self._generator_optimizer = optimizer_generator
        self._discriminator_loss_fn = loss_discriminator
        self._generator_loss_fn = loss_generator

    def gradient_penalty(self, batch_size, real_feature, real_label, synthetic_feature):
        """
        Compute the gradient penalty for the WassersteinGP GAN.

        The gradient penalty is used to enforce the Lipschitz constraint on the discriminator's output.

        Parameters:
            batch_size (int):
                The batch size of the input data.
            real_feature:
                Real data features.
            real_label:
                Real data labels.
            synthetic_feature:
                Synthetic (generated) data features.

        Returns:
            Gradient penalty value.
        """
        if self._framework == 'tensorflow':
            return self._gradient_penalty_tensorflow(batch_size, real_feature, real_label, synthetic_feature)
        else:
            return self._gradient_penalty_pytorch(batch_size, real_feature, real_label, synthetic_feature)

    def _gradient_penalty_tensorflow(self, batch_size, real_feature, real_label, synthetic_feature):
        """TensorFlow implementation of gradient penalty."""
        random_smooth = self.tf.random.normal([batch_size, 1], 0.0, 0.1)
        linear_distance = synthetic_feature - real_feature
        interpolated_feature = real_feature + random_smooth * linear_distance

        with self.tf.GradientTape() as gradient_penalty:
            gradient_penalty.watch(interpolated_feature)
            labels_predicted = self._discriminator([interpolated_feature, real_label], training=True)

        gradient_computed = gradient_penalty.gradient(labels_predicted, [interpolated_feature])[0]
        gradient_normalized = self.tf.sqrt(self.tf.reduce_sum(self.tf.square(gradient_computed), axis=[1]))
        gradient_penalty_final = self.tf.reduce_mean((gradient_normalized - 1.0) ** 2)

        return gradient_penalty_final

    def _gradient_penalty_pytorch(self, batch_size, real_feature, real_label, synthetic_feature):
        """PyTorch implementation of gradient penalty."""
        alpha = self.torch.randn(batch_size, 1, device=self._device) * 0.1
        
        # Ensure proper broadcasting
        while alpha.dim() < real_feature.dim():
            alpha = alpha.unsqueeze(-1)
        
        interpolated_feature = real_feature + alpha * (synthetic_feature - real_feature)
        interpolated_feature.requires_grad_(True)

        labels_predicted = self._discriminator(interpolated_feature, real_label)

        gradients = self.torch.autograd.grad(
            outputs=labels_predicted,
            inputs=interpolated_feature,
            grad_outputs=self.torch.ones_like(labels_predicted),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty_final = self.torch.mean((gradient_norm - 1.0) ** 2)

        return gradient_penalty_final

    def train_step(self, batch):
        """
        Executes one training step for the GAN model.

        This step updates both the discriminator and the generator.
        The discriminator is updated multiple times (controlled by self._discriminator_steps),
        while the generator is updated once.

        Args:
            batch (tuple): A tuple containing:
                - real_feature: A batch of real data samples (features).
                - real_samples_label: Corresponding class labels for each sample.

        Returns:
            dict: Dictionary containing the discriminator and generator loss for the current training step.
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

        # Discriminator Training Loop
        for _ in range(self._discriminator_steps):
            latent_space = self.tf.random.normal((batch_size, self._latent_dimension),
                                                 mean=self._latent_mean_distribution,
                                                 stddev=self._latent_stander_deviation)

            with self.tf.GradientTape() as discriminator_gradient:
                synthetic_feature = self._generator([latent_space, real_samples_label], training=False)
                label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
                label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

                discriminator_loss_result = self._discriminator_loss_fn(
                    real_img=label_predicted_real, fake_img=label_predicted_synthetic)

                gradient_penalty = self.gradient_penalty(batch_size, real_feature, 
                                                        real_samples_label, synthetic_feature)

                all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            gradient_computed = discriminator_gradient.gradient(all_discriminator_loss,
                                                               self._discriminator.trainable_variables)
            self._discriminator_optimizer.apply_gradients(zip(gradient_computed,
                                                             self._discriminator.trainable_variables))

        # Generator Training Step
        latent_space = self.tf.random.normal((batch_size, self._latent_dimension),
                                            mean=self._latent_mean_distribution,
                                            stddev=self._latent_stander_deviation)

        with self.tf.GradientTape() as generator_gradient:
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)
            all_generator_loss = self._generator_loss_fn(predicted_labels)

        gradient_computed = generator_gradient.gradient(all_generator_loss, self._generator.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(gradient_computed, self._generator.trainable_variables))

        self._d_loss_tracker.update_state(all_discriminator_loss)
        self._g_loss_tracker.update_state(all_generator_loss)

        return {"d_loss": self._d_loss_tracker.result(), "g_loss": self._g_loss_tracker.result()}

    def _train_step_pytorch(self, batch):
        """PyTorch implementation of train_step."""
        real_feature, real_samples_label = batch
        real_feature = real_feature.to(self._device)
        real_samples_label = real_samples_label.to(self._device)
        
        batch_size = real_feature.size(0)
        
        if real_samples_label.dim() == 1:
            real_samples_label = real_samples_label.unsqueeze(-1)

        # Discriminator Training Loop
        for _ in range(self._discriminator_steps):
            self._discriminator_optimizer.zero_grad()

            latent_space = self.torch.randn(batch_size, self._latent_dimension, device=self._device)
            latent_space = latent_space * self._latent_stander_deviation + self._latent_mean_distribution

            with self.torch.no_grad():
                synthetic_feature = self._generator(latent_space, real_samples_label)

            label_predicted_real = self._discriminator(real_feature, real_samples_label)
            label_predicted_synthetic = self._discriminator(synthetic_feature, real_samples_label)

            discriminator_loss_result = self._discriminator_loss_fn(
                real_img=label_predicted_real, fake_img=label_predicted_synthetic)

            gradient_penalty = self.gradient_penalty(batch_size, real_feature, 
                                                    real_samples_label, synthetic_feature)

            all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            all_discriminator_loss.backward()
            self._discriminator_optimizer.step()

        # Generator Training Step
        self._generator_optimizer.zero_grad()

        latent_space = self.torch.randn(batch_size, self._latent_dimension, device=self._device)
        latent_space = latent_space * self._latent_stander_deviation + self._latent_mean_distribution

        synthetic_feature = self._generator(latent_space, real_samples_label)

        with self.torch.no_grad():
            self._discriminator.eval()
        
        predicted_labels = self._discriminator(synthetic_feature, real_samples_label)
        self._discriminator.train()

        all_generator_loss = self._generator_loss_fn(predicted_labels)

        all_generator_loss.backward()
        self._generator_optimizer.step()

        self._d_loss_total += all_discriminator_loss.item()
        self._g_loss_total += all_generator_loss.item()
        self._num_batches += 1

        return {
            "d_loss": self._d_loss_total / self._num_batches, 
            "g_loss": self._g_loss_total / self._num_batches
        }

    def fit(self, train_dataset, epochs=1, verbose=1):
        """
        Trains the WGAN-GP model.

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
                print(f"Epoch {epoch+1}/{epochs} - d_loss: {avg_d_loss:.4f} - g_loss: {avg_g_loss:.4f}")

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to the number of samples to generate for each class.
                - "number_classes" (int): Total number of classes (used for one-hot encoding).

        Returns:
            dict: A dictionary where each key is a class label and the value is an array of generated samples.
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
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                    num_classes=number_samples_per_class["number_classes"])

            latent_noise = numpy.random.normal(loc=self._latent_mean_distribution,
                                              scale=self._latent_stander_deviation,
                                              size=(number_instances, self._latent_dimension))

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
                ) * self._latent_stander_deviation + self._latent_mean_distribution

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

    # Properties
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
    def gradient_penalty_weight(self):
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value):
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

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
    def latent_stander_deviation(self):
        return self._latent_stander_deviation

    @latent_stander_deviation.setter
    def latent_stander_deviation(self, value):
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_stander_deviation = value

    @property
    def file_name_discriminator(self):
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value):
        self._file_name_discriminator = value

    @property
    def file_name_generator(self):
        return self._file
