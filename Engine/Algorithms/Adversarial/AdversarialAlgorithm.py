#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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
    import numpy
    import logging
    from pathlib import Path

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class AdversarialAlgorithm:
    """
    Implements an adversarial training algorithm, typically used in Generative Adversarial Networks (GANs).

    This class performs adversarial training by utilizing a generator and a discriminator,
    optimizing the generator to produce realistic data while training the discriminator to differentiate
    between real and fake data.

    The concept of Generative Adversarial Networks was introduced by Ian Goodfellow and his collaborators.

    Attributes:
        @framework (str):
            Framework to use: 'tensorflow' or 'pytorch'.
        @generator_model (Model):
            The generator model.
        @discriminator_model (Model):
            The discriminator model.
        @latent_dimension (int):
            Dimensionality of the latent space.
        @loss_generator (function):
            Loss function for the generator.
        @loss_discriminator (function):
            Loss function for the discriminator.
        @file_name_discriminator (str):
            Filename for saving the discriminator model.
        @file_name_generator (str):
            Filename for saving the generator model.
        @models_saved_path (str):
            Path where models will be saved.
        @latent_mean_distribution (float):
            Mean of the latent noise distribution.
        @latent_stander_deviation (float):
            Standard deviation of the latent noise distribution.
        @smoothing_rate (float):
            Smoothing rate applied to discriminator labels.

    Example:
        >>> # TensorFlow example
        >>> adversarial_algorithm = AdversarialAlgorithm(
        ...     framework='tensorflow',
        ...     generator_model=generator_model,
        ...     discriminator_model=discriminator_model,
        ...     latent_dimension=100,
        ...     loss_generator=tf.keras.losses.BinaryCrossEntropy(),
        ...     loss_discriminator=tf.keras.losses.BinaryCrossEntropy(),
        ...     file_name_discriminator="discriminator.h5",
        ...     file_name_generator="generator.h5",
        ...     models_saved_path="./models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1
        ... )
        >>> # PyTorch example
        >>> adversarial_algorithm = AdversarialAlgorithm(
        ...     framework='pytorch',
        ...     generator_model=generator_model,
        ...     discriminator_model=discriminator_model,
        ...     latent_dimension=100,
        ...     loss_generator=nn.BCELoss(),
        ...     loss_discriminator=nn.BCELoss(),
        ...     file_name_discriminator="discriminator.pt",
        ...     file_name_generator="generator.pt",
        ...     models_saved_path="./models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1
        ... )
    """

    def __init__(self, 
                 framework,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 loss_generator,
                 loss_discriminator,
                 file_name_discriminator,
                 file_name_generator,
                 models_saved_path,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 smoothing_rate,
                 *args,
                 **kwargs):
        """
        Initializes the adversarial algorithm with the specified generator, discriminator, and other configurations.

        Args:
            @framework (str):
                Framework to use: 'tensorflow' or 'pytorch'.
            @generator_model (Model):
                The generator model.
            @discriminator_model (Model):
                The discriminator model.
            @latent_dimension (int):
                Latent space dimension.
            @loss_generator (function):
                Generator's loss function.
            @loss_discriminator (function):
                Discriminator's loss function.
            @file_name_discriminator (str):
                Filename for discriminator model.
            @file_name_generator (str):
                Filename for generator model.
            @models_saved_path (str):
                Path for saving models.
            @latent_mean_distribution (float):
                Mean of the latent noise distribution.
            @latent_stander_deviation (float):
                Standard deviation of the latent noise.
            @smoothing_rate (float):
                Label smoothing rate.
            @*args, **kwargs:
                Additional arguments.
        """

        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")

        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")

        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("Models saved path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")

        if not isinstance(latent_stander_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")

        if latent_stander_deviation <= 0:
            raise ValueError("Latent standard deviation must be greater than 0.")

        if not (0.0 <= smoothing_rate <= 1.0):
            raise ValueError("Smoothing rate must be between 0 and 1.")

        self._framework = framework
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._optimizer_generator = None
        self._optimizer_discriminator = None
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator
        self._smoothing_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path

        # Import framework-specific modules
        if self._framework == 'tensorflow':
            import tensorflow as tf
            self.tf = tf
        else:  # pytorch
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._generator.to(self._device)
            self._discriminator.to(self._device)

    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, *args,
                **kwargs):
        """
        Compiles the adversarial algorithm by setting optimizers and loss functions for both generator and discriminator.

        Args:
            optimizer_generator (Optimizer): Optimizer for the generator.
            optimizer_discriminator (Optimizer): Optimizer for the discriminator.
            loss_generator (function): Generator's loss function.
            loss_discriminator (function): Discriminator's loss function.
            *args, **kwargs: Additional arguments.
        """
        self._optimizer_generator = optimizer_generator
        self._optimizer_discriminator = optimizer_discriminator
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator

    def train_step(self, batch):
        """
        Performs a single training step for both generator and discriminator.

        Args:
            batch (tuple): A tuple containing real features (input data) and their corresponding labels.

        Returns:
            dict: A dictionary containing the loss values for both generator (loss_g) and discriminator (loss_d).
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

        latent_space = self.tf.random.normal(shape=(batch_size, self._latent_dimension))
        synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

        with self.tf.GradientTape() as discriminator_gradient:
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)
            label_predicted_all_samples = self.tf.concat([label_predicted_real, label_predicted_synthetic], axis=0)

            list_all_labels_predicted = [
                self.tf.zeros_like(label_predicted_real),
                self.tf.ones_like(label_predicted_synthetic)
            ]
            tensor_labels_predicted = self.tf.concat(list_all_labels_predicted, axis=0)

            smooth_tensor_real_data = 0.15 * self.tf.random.uniform(self.tf.shape(label_predicted_real))
            smooth_tensor_synthetic_data = -0.15 * self.tf.random.uniform(
                self.tf.shape(label_predicted_synthetic))

            tensor_labels_predicted += self.tf.concat(
                [smooth_tensor_real_data, smooth_tensor_synthetic_data], axis=0
            )

            loss_value = self._loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        gradient_tape_loss = discriminator_gradient.gradient(loss_value, self._discriminator.trainable_variables)
        self._optimizer_discriminator.apply_gradients(zip(gradient_tape_loss, self._discriminator.trainable_variables))

        with self.tf.GradientTape() as generator_gradient:
            latent_space = self.tf.random.normal(shape=(batch_size, self._latent_dimension))
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)
            total_loss_g = self._loss_generator(self.tf.zeros_like(predicted_labels), predicted_labels)

        gradient_tape_loss = generator_gradient.gradient(total_loss_g, self._generator.trainable_variables)
        self._optimizer_generator.apply_gradients(zip(gradient_tape_loss, self._generator.trainable_variables))

        return {"loss_d": loss_value, "loss_g": total_loss_g}

    def _train_step_pytorch(self, batch):
        """PyTorch implementation of train_step."""
        real_feature, real_samples_label = batch
        real_feature = real_feature.to(self._device)
        real_samples_label = real_samples_label.to(self._device)
        
        batch_size = real_feature.shape[0]
        
        if len(real_samples_label.shape) == 1:
            real_samples_label = real_samples_label.unsqueeze(-1)

        # Train Discriminator
        self._optimizer_discriminator.zero_grad()
        
        latent_space = self.torch.randn(batch_size, self._latent_dimension, device=self._device)
        with self.torch.no_grad():
            synthetic_feature = self._generator(latent_space, real_samples_label)

        label_predicted_real = self._discriminator(real_feature, real_samples_label)
        label_predicted_synthetic = self._discriminator(synthetic_feature, real_samples_label)
        label_predicted_all_samples = self.torch.cat([label_predicted_real, label_predicted_synthetic], dim=0)

        tensor_labels_predicted = self.torch.cat([
            self.torch.zeros_like(label_predicted_real),
            self.torch.ones_like(label_predicted_synthetic)
        ], dim=0)

        smooth_tensor_real_data = 0.15 * self.torch.rand_like(label_predicted_real)
        smooth_tensor_synthetic_data = -0.15 * self.torch.rand_like(label_predicted_synthetic)
        tensor_labels_predicted += self.torch.cat([smooth_tensor_real_data, smooth_tensor_synthetic_data], dim=0)

        loss_value = self._loss_discriminator(label_predicted_all_samples, tensor_labels_predicted)
        loss_value.backward()
        self._optimizer_discriminator.step()

        # Train Generator
        self._optimizer_generator.zero_grad()
        
        latent_space = self.torch.randn(batch_size, self._latent_dimension, device=self._device)
        synthetic_feature = self._generator(latent_space, real_samples_label)
        predicted_labels = self._discriminator(synthetic_feature, real_samples_label)
        
        total_loss_g = self._loss_generator(predicted_labels, self.torch.zeros_like(predicted_labels))
        total_loss_g.backward()
        self._optimizer_generator.step()

        return {
            "loss_d": loss_value.item() if hasattr(loss_value, 'item') else float(loss_value),
            "loss_g": total_loss_g.item() if hasattr(total_loss_g, 'item') else float(total_loss_g)
        }

    def fit(self, train_dataset, epochs=1, verbose=1):
        """
        Trains the adversarial model.

        Args:
            train_dataset: Training dataset.
            epochs (int): Number of epochs.
            verbose (int): Verbosity mode.
        """
        for epoch in range(epochs):
            epoch_loss_d = []
            epoch_loss_g = []
            
            for batch in train_dataset:
                losses = self.train_step(batch)
                epoch_loss_d.append(losses['loss_d'])
                epoch_loss_g.append(losses['loss_g'])
            
            if verbose:
                avg_loss_d = numpy.mean(epoch_loss_d)
                avg_loss_g = numpy.mean(epoch_loss_g)
                logging.info(f"Epoch {epoch+1}/{epochs} - loss_d: {avg_loss_d:.4f} - loss_g: {avg_loss_g:.4f}")

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict):
                A dictionary specifying the number of synthetic samples to generate per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples for that class.
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
                # Create one-hot encoded labels
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

    def save_model(self, path_output, k_fold):
        """Saves the generator and discriminator models."""
        if self._framework == 'tensorflow':
            self._save_model_tensorflow(path_output, k_fold)
        else:
            self._save_model_pytorch(path_output, k_fold)

    def _save_model_tensorflow(self, path_output, k_fold):
        """TensorFlow implementation of save_model."""
        try:
            logging.info("Starting to save Adversarial Model for fold {}...".format(k_fold))

            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified directory at: {}".format(path_directory))

            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified fold directory at: {}".format(path_model))

            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            logging.info("Saving discriminator model...")
            discriminator_model_json = self._discriminator.to_json()

            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self._discriminator.save_weights(discriminator_file_name + ".h5")
            logging.info("Discriminator model saved at: {}.json and {}.h5".format(discriminator_file_name,
                                                                                  discriminator_file_name))

            logging.info("Saving generator model...")
            generator_model_json = self._generator.to_json()

            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self._generator.save_weights(generator_file_name + ".h5")
            logging.info("Generator model saved at: {}.json and {}.h5".format(generator_file_name,
                                                                              generator_file_name))

        except FileExistsError:
            logging.error("Model file already exists. Aborting.")
            exit(-1)
        except Exception as e:
            logging.error("An error occurred while saving the models: {}".format(e))
            exit(-1)

    def _save_model_pytorch(self, path_output, k_fold):
        """PyTorch implementation of save_model."""
        try:
            logging.info("Starting to save Adversarial Model for fold {}...".format(k_fold))

            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified directory at: {}".format(path_directory))

            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified fold directory at: {}".format(path_model))

            discriminator_file_name = os.path.join(path_model, discriminator_file_name + ".pt")
            generator_file_name = os.path.join(path_model, generator_file_name + ".pt")

            logging.info("Saving discriminator model...")
            self.torch.save({
                'model_state_dict': self._discriminator.state_dict(),
                'optimizer_state_dict': self._optimizer_discriminator.state_dict() if self._optimizer_discriminator else None
            }, discriminator_file_name)
            logging.info("Discriminator model saved at: {}".format(discriminator_file_name))

            logging.info("Saving generator model...")
            self.torch.save({
                'model_state_dict': self._generator.state_dict(),
                'optimizer_state_dict': self._optimizer_generator.state_dict() if self._optimizer_generator else None
            }, generator_file_name)
            logging.info("Generator model saved at: {}".format(generator_file_name))

        except Exception as e:
            logging.error("An error occurred while saving the models: {}".format(e))
            exit(-1)

    def load_models(self, path_output, k_fold):
        """Loads the generator and discriminator models."""
        if self._framework == 'tensorflow':
            self._load_models_tensorflow(path_output, k_fold)
        else:
            self._load_models_pytorch(path_output, k_fold)

    def _load_models_tensorflow(self, path_output, k_fold):
        """TensorFlow implementation of load_models."""
        from tensorflow.keras.models import model_from_json
        
        try:
            logging.info("Loading Adversarial Model for fold {}...".format(k_fold + 1))

            path_directory = os.path.join(path_output, self._models_saved_path)

            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold + 1)
            generator_file_name = self._file_name_generator + "_" + str(k_fold + 1)

            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            logging.info("Loading discriminator model from: {}.json".format(discriminator_file_name))
            with open(discriminator_file_name + ".json", 'r') as json_file:
                discriminator_model_json = json_file.read()

            self._discriminator = model_from_json(discriminator_model_json)
            self._discriminator.load_weights(discriminator_file_name + ".h5")
            logging.info("Loaded discriminator weights from: {}.h5".format(discriminator_file_name))

            logging.info("Loading generator model from: {}.json".format(generator_file_name))
            with open(generator_file_name + ".json", 'r') as json_file:
                generator_model_json = json_file.read()

            self._generator = model_from_json(generator_model_json)
            self._generator.load_weights(generator_file_name + ".h5")
            logging.info("Loaded generator weights from: {}.h5".format(generator_file_name))

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            exit(-1)
        except Exception as e:
            logging.error("An error occurred while loading the models: {}".format(e))
            exit(-1)

    def _load_models_pytorch(self, path_output, k_fold):
        """PyTorch implementation of load_models."""
        try:
            logging.info("Loading Adversarial Model for fold {}...".format(k_fold + 1))

            path_directory = os.path.join(path_output, self._models_saved_path)

            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold + 1) + ".pt"
            generator_file_name = self._file_name_generator + "_" + str(k_fold + 1) + ".pt"

            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            logging.info("Loading discriminator model from: {}".format(discriminator_file_name))
            checkpoint = self.torch.load(discriminator_file_name, map_location=self._device)
            self._discriminator.load_state_dict(checkpoint['model_state_dict'])
            if self._optimizer_discriminator and checkpoint.get('optimizer_state_dict'):
                self._optimizer_discriminator.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("Loaded discriminator from: {}".format(discriminator_file_name))

            logging.info("Loading generator model from: {}".format(generator_file_name))
            checkpoint = self.torch.load(generator_file_name, map_location=self._device)
            self._generator.load_state_dict(checkpoint['model_state_dict'])
            if self._optimizer_generator and checkpoint.get('optimizer_state_dict'):
                self._optimizer_generator.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("Loaded generator from: {}".format(generator_file_name))

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            exit(-1)
        except Exception as e:
            logging.error("An error occurred while loading the models: {}".format(e))
            exit(-1)

    def set_generator(self, generator):
        self._generator = generator
        if self._framework == 'pytorch':
            self._generator.to(self._device)

    def set_discriminator(self, discriminator):
        self._discriminator = discriminator
        if self._framework == 'pytorch':
            self._discriminator.to(self._device)

    def set_latent_dimension(self, latent_dimension):
        self._latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        self._optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        self._optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        self._loss_generator = loss_generator

    def set_loss_discriminator(self, loss_discriminator):
        self._loss_discriminator = loss_discriminator
