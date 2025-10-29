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


class QuantizedVAEAlgorithm:
    """
    Implements a Vector Quantized Variational Autoencoder (VQ-VAE) for discrete latent
    representation learning and generation supporting both TensorFlow and PyTorch.
    
    This model combines an encoder, decoder, and vector quantization layer to learn 
    compressed representations of input data.

    The algorithm supports training with reconstruction loss and commitment loss for
    the quantization layer, enabling stable training of discrete latent variables.

    Attributes:
        @framework (str):
            Framework to use: 'tensorflow' or 'pytorch'.
        @train_variance (float):
            Variance of the training data used to normalize the reconstruction loss.
        @latent_dimension (int):
            Dimensionality of the latent space before quantization.
        @number_embeddings (int):
            Number of embeddings in the vector quantization codebook.
        @encoder (Model):
            Encoder network that maps input data to latent representations.
        @decoder (Model):
            Decoder network that reconstructs data from quantized latent codes.
        @quantized_vae_model (Model):
            Complete VQ-VAE model combining encoder, quantization, and decoder.
        @file_name_encoder (str):
            Filename for saving the encoder weights.
        @file_name_decoder (str):
            Filename for saving the decoder weights.
        @models_saved_path (str):
            Directory path for saving model weights.
        @total_loss_tracker:
            Metric tracker for total training loss (reconstruction + VQ losses).
        @reconstruction_loss_tracker:
            Metric tracker for reconstruction loss component.
        @vq_loss_tracker:
            Metric tracker for vector quantization loss components.

    References:
        - van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). "Neural Discrete
        Representation Learning." Advances in Neural Information Processing Systems (NeurIPS).

    Example:
        >>> # TensorFlow example
        >>> vae = QuantizedVAEAlgorithm(
        ...     framework='tensorflow',
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     quantized_vae_model=vq_vae,
        ...     train_variance=0.1,
        ...     latent_dimension=64,
        ...     number_embeddings=512,
        ...     file_name_encoder="encoder_weights.h5",
        ...     file_name_decoder="decoder_weights.h5",
        ...     models_saved_path="./saved_models/"
        ... )
        >>> # PyTorch example
        >>> vae = QuantizedVAEAlgorithm(
        ...     framework='pytorch',
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     quantized_vae_model=vq_vae,
        ...     train_variance=0.1,
        ...     latent_dimension=64,
        ...     number_embeddings=512,
        ...     file_name_encoder="encoder_weights.pt",
        ...     file_name_decoder="decoder_weights.pt",
        ...     models_saved_path="./saved_models/"
        ... )
    """

    def __init__(self,
                 framework,
                 encoder_model,
                 decoder_model,
                 quantized_vae_model,
                 train_variance,
                 latent_dimension,
                 number_embeddings,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path,
                 **kwargs):
        """
        Initializes the QuantizedVAEAlgorithm with encoder, decoder, and VQ-VAE components.

        Args:
            @framework (str):
                Framework to use: 'tensorflow' or 'pytorch'.
            @encoder_model (Model):
                Encoder network that compresses input data to latent space.
            @decoder_model (Model):
                Decoder network that reconstructs data from quantized latent codes.
            @quantized_vae_model (Model):
                Complete VQ-VAE model including quantization layer.
            @train_variance (float):
                Data variance used to scale reconstruction loss.
            @latent_dimension (int):
                Dimensionality of latent space before quantization.
            @number_embeddings (int):
                Size of quantization codebook (number of discrete latent codes).
            @file_name_encoder (str):
                Filename for saving encoder weights.
            @file_name_decoder (str):
                Filename for saving decoder weights.
            @models_saved_path (str):
                Directory path for model weight storage.
            @**kwargs:
                Additional keyword arguments.

        Raises:
            ValueError:
                If framework is not 'tensorflow' or 'pytorch'.
                If latent_dimension <= 0.
                If number_embeddings <= 0.
                If train_variance <= 0.
        """

        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        if latent_dimension <= 0:
            raise ValueError("latent_dimension must be greater than 0.")

        if number_embeddings <= 0:
            raise ValueError("number_embeddings must be greater than 0.")

        if train_variance <= 0:
            raise ValueError("train_variance must be greater than 0.")

        self._framework = framework
        self._train_variance = train_variance
        self._latent_dimension = latent_dimension
        self._number_embeddings = number_embeddings
        self._encoder = encoder_model
        self._decoder = decoder_model
        self._quantized_vae_model = quantized_vae_model
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder
        self._models_saved_path = models_saved_path
        self.optimizer = None

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            from tensorflow.keras.metrics import Mean
            
            self.tf = tf
            self.total_loss_tracker = Mean(name="total_loss")
            self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
            self.vq_loss_tracker = Mean(name="vq_loss")
            
        else:  # pytorch
            import torch
            import torch.nn as nn
            
            self.torch = torch
            self.nn = nn
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._encoder.to(self._device)
            self._decoder.to(self._device)
            self._quantized_vae_model.to(self._device)
            
            # Manual loss tracking for PyTorch
            self._total_loss = 0.0
            self._reconstruction_loss = 0.0
            self._vq_loss = 0.0
            self._num_batches = 0

    @property
    def metrics(self):
        """
        Returns the list of metrics tracked during training.

        Returns:
            list: List of metric trackers [total_loss, reconstruction_loss, vq_loss].
        """
        if self._framework == 'tensorflow':
            return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.vq_loss_tracker]
        else:
            # For PyTorch, return dict-like objects with result() method
            class MetricTracker:
                def __init__(self, name, value):
                    self.name = name
                    self._value = value
                
                def result(self):
                    return self._value
            
            return [
                MetricTracker("total_loss", self._total_loss / max(self._num_batches, 1)),
                MetricTracker("reconstruction_loss", self._reconstruction_loss / max(self._num_batches, 1)),
                MetricTracker("vq_loss", self._vq_loss / max(self._num_batches, 1))
            ]

    def compile(self, optimizer, *args, **kwargs):
        """
        Compiles the VQ-VAE model with an optimizer.

        Args:
            optimizer: Optimizer for training.
            *args, **kwargs: Additional arguments.
        """
        self.optimizer = optimizer

    def train_step(self, data):
        """
        Performs a single training step on a batch of data.

        The training step includes:
        1. Forward pass through the VQ-VAE
        2. Loss computation (reconstruction + VQ losses)
        3. Gradient computation and weight updates

        Args:
            data (tuple): Input data tuple containing (input_tensor, labels).

        Returns:
            dict: Dictionary with loss metrics for the current step.
        """
        if self._framework == 'tensorflow':
            return self._train_step_tensorflow(data)
        else:
            return self._train_step_pytorch(data)

    def _train_step_tensorflow(self, data):
        """TensorFlow implementation of train_step."""
        x, y = data
        output_tensor, _ = x

        with self.tf.GradientTape() as tape:
            reconstructions = self._quantized_vae_model(x)
            reconstruction_loss = (self.tf.reduce_mean((output_tensor - reconstructions) ** 2) / self._train_variance)
            vae_model_loss = reconstruction_loss + sum(self._quantized_vae_model.losses)

        gradient_flow = tape.gradient(vae_model_loss, self._quantized_vae_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_flow, self._quantized_vae_model.trainable_variables))

        self.total_loss_tracker.update_state(vae_model_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self._quantized_vae_model.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    def _train_step_pytorch(self, data):
        """PyTorch implementation of train_step."""
        x, y = data
        
        # Handle input tensor format
        if isinstance(x, (list, tuple)):
            output_tensor = x[0].to(self._device)
            x = [item.to(self._device) if hasattr(item, 'to') else item for item in x]
        else:
            output_tensor = x.to(self._device)
            x = x.to(self._device)

        self.optimizer.zero_grad()
        
        reconstructions = self._quantized_vae_model(x)
        reconstruction_loss = self.torch.mean((output_tensor - reconstructions) ** 2) / self._train_variance
        
        # Get VQ losses (commitment loss and codebook loss)
        vq_loss = 0.0
        if hasattr(self._quantized_vae_model, 'vq_loss'):
            vq_loss = self._quantized_vae_model.vq_loss
        
        vae_model_loss = reconstruction_loss + vq_loss
        vae_model_loss.backward()
        self.optimizer.step()

        # Update tracked metrics
        self._total_loss += vae_model_loss.item()
        self._reconstruction_loss += reconstruction_loss.item()
        self._vq_loss += vq_loss.item() if isinstance(vq_loss, self.torch.Tensor) else vq_loss
        self._num_batches += 1

        return {
            "loss": self._total_loss / self._num_batches,
            "reconstruction_loss": self._reconstruction_loss / self._num_batches,
            "vqvae_loss": self._vq_loss / self._num_batches,
        }

    def fit(self, train_dataset, epochs=1, verbose=1):
        """
        Trains the VQ-VAE model.

        Args:
            train_dataset: Training dataset.
            epochs (int): Number of epochs.
            verbose (int): Verbosity mode.
        """
        for epoch in range(epochs):
            if self._framework == 'pytorch':
                self._total_loss = 0.0
                self._reconstruction_loss = 0.0
                self._vq_loss = 0.0
                self._num_batches = 0
            
            epoch_results = {'loss': [], 'reconstruction_loss': [], 'vqvae_loss': []}
            
            for batch in train_dataset:
                result = self.train_step(batch)
                for key in epoch_results:
                    epoch_results[key].append(float(result[key]))
            
            if verbose:
                avg_loss = numpy.mean(epoch_results['loss'])
                avg_recon = numpy.mean(epoch_results['reconstruction_loss'])
                avg_vq = numpy.mean(epoch_results['vqvae_loss'])
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
                      f"reconstruction_loss: {avg_recon:.4f} - vqvae_loss: {avg_vq:.4f}")

    def get_samples(self, number_samples_per_class):
        """
        Generates samples from the latent space using the decoder.

        Samples are generated by:
        1. Randomly selecting indices from the codebook
        2. Gathering corresponding latent vectors
        3. Decoding these vectors conditioned on class labels

        Args:
            number_samples_per_class (dict):
                Dictionary specifying samples to generate per class with structure:
                {
                    "classes": {class_label: num_samples, ...},
                    "number_classes": total_num_classes
                }

        Returns:
            dict: Generated samples keyed by class label.
        """
        if self._framework == 'tensorflow':
            return self._get_samples_tensorflow(number_samples_per_class)
        else:
            return self._get_samples_pytorch(number_samples_per_class)

    def _get_samples_tensorflow(self, number_samples_per_class):
        """TensorFlow implementation of get_samples."""
        from tensorflow.keras.utils import to_categorical
        
        generated_data = {}
        codebook = self._quantized_vae_model.get_layer("vector_quantizer").embeddings
        number_embeddings = self._number_embeddings

        for label_class, number_instances in number_samples_per_class["classes"].items():
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            sampled_indices = numpy.random.choice(number_embeddings, size=number_instances)
            quantized_vectors = self.tf.gather(codebook, sampled_indices)
            quantized_vectors = self.tf.convert_to_tensor(quantized_vectors)

            generated_samples = self._decoder.predict([quantized_vectors, label_samples_generated], verbose=0)
            generated_samples = numpy.rint(generated_samples)
            generated_data[label_class] = generated_samples

        return generated_data

    def _get_samples_pytorch(self, number_samples_per_class):
        """PyTorch implementation of get_samples."""
        generated_data = {}
        self._decoder.eval()

        # Get codebook from the VQ layer
        codebook = None
        for name, module in self._quantized_vae_model.named_modules():
            if 'vector_quantizer' in name.lower() or 'vq' in name.lower():
                if hasattr(module, 'embeddings') or hasattr(module, 'embedding'):
                    codebook = getattr(module, 'embeddings', getattr(module, 'embedding', None))
                    break
        
        if codebook is None:
            raise ValueError("Could not find codebook embeddings in quantized_vae_model")

        with self.torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                label_samples_generated = self.torch.zeros(
                    number_instances,
                    number_samples_per_class["number_classes"],
                    device=self._device
                )
                label_samples_generated[:, label_class] = 1

                sampled_indices = numpy.random.choice(self._number_embeddings, size=number_instances)
                sampled_indices = self.torch.tensor(sampled_indices, device=self._device)
                
                if isinstance(codebook, self.torch.Tensor):
                    quantized_vectors = codebook[sampled_indices]
                else:
                    quantized_vectors = self.torch.index_select(codebook.weight, 0, sampled_indices)

                generated_samples = self._decoder(quantized_vectors, label_samples_generated)
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
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder
        if self._framework == 'pytorch':
            self._encoder.to(self._device)

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
        if self._framework == 'pytorch':
            self._decoder.to(self._device)
