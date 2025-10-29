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
    import sys
    import numpy

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_DIFFUSION_GAUSSIAN_BETA_START = 1e-4
DEFAULT_DIFFUSION_GAUSSIAN_BETA_END = 0.02
DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS = 1000
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN = -1.0
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX = 1.0

class GaussianDiffusion:
    """
    A framework-agnostic class representing the Gaussian diffusion process used in diffusion 
    models for denoising and generative tasks. Supports both TensorFlow and PyTorch.

    This implementation follows the method proposed by Ho et al. (2020), where a sequence of
    Gaussian noise is applied iteratively to a data sample, and a neural network is trained
    to reverse this process, allowing the generation of high-quality synthetic samples.

    Reference:
    ----------
        Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
        Advances in Neural Information Processing Systems (NeurIPS).


    Mathematical Formalism:
    -----------------------
        The diffusion process follows these key equations:

        1. Forward Process (q):
            q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)
            where β_t is the noise schedule

        2. Cumulative Forward Process:
            q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
            where α_t = 1-β_t and ᾱ_t = ∏_{s=1}^t α_s

        3. Reverse Process (p_θ):
            p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

        4. Posterior Distribution (q):
            q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_tI)
            where:
            μ̃_t(x_t,x_0) = (√ᾱ_{t-1}β_t)/(1-ᾱ_t)x_0 + (√α_t(1-ᾱ_{t-1}))/(1-ᾱ_t)x_t
            β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t)β_t


    Attributes:
    -----------
        @framework: str
            Framework to use: 'tensorflow' or 'pytorch'.
        @beta_start: float
            The starting value of the beta schedule, controlling noise variance.
        @beta_end: float
            The ending value of the beta schedule, defining the final noise level.
        @time_steps: int
            The number of discrete time steps for the diffusion process.
        @clip_min: float
            The minimum value to clip the output during denoising, ensuring numerical stability.
        @clip_max: float
            The maximum value to clip the output during denoising.

    Example:
        >>> # TensorFlow example
        >>> diffusion = GaussianDiffusion(
        ...     framework='tensorflow',
        ...     beta_start=0.0001, 
        ...     beta_end=0.02, 
        ...     time_steps=1000, 
        ...     clip_min=-1.0, 
        ...     clip_max=1.0
        ... )
        >>> noise = tf.random.normal(shape=(1, 32, 32, 3))
        >>> t = tf.constant([10], dtype=tf.int32)
        >>> x_t = diffusion.q_sample(x_start=noise, t=t, noise=noise)
        >>> print(x_t.shape)  # Expected output: (1, 32, 32, 3)
        
        >>> # PyTorch example
        >>> diffusion = GaussianDiffusion(
        ...     framework='pytorch',
        ...     beta_start=0.0001, 
        ...     beta_end=0.02, 
        ...     time_steps=1000, 
        ...     clip_min=-1.0, 
        ...     clip_max=1.0
        ... )
        >>> noise = torch.randn(1, 3, 32, 32)
        >>> t = torch.tensor([10], dtype=torch.long)
        >>> x_t = diffusion.q_sample(x_start=noise, t=t, noise=noise)
        >>> print(x_t.shape)  # Expected output: torch.Size([1, 3, 32, 32])

    """
    def __init__(self,
                 framework,
                 beta_start: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_START,
                 beta_end: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_END,
                 time_steps: float = DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS,
                 clip_min: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN,
                 clip_max: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX):
        """
        Initializes the GaussianDiffusion class with the given beta schedule and clipping values.

        Parameters:
        -----------
            framework : str
                Framework to use: 'tensorflow' or 'pytorch'.
            beta_start : float
                Starting value for beta.
            beta_end : float
                Ending value for beta.
            time_steps : int
                Number of time steps in the diffusion process.
            clip_min : float
                Minimum value for clipping the denoised output.
            clip_max : float
                Maximum value for clipping the denoised output.
        """
        
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError("Framework must be either 'tensorflow' or 'pytorch'.")

        self._framework = framework
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._time_steps = time_steps
        self._clip_min = clip_min
        self._clip_max = clip_max

        # Framework-specific initialization
        if self._framework == 'tensorflow':
            import tensorflow as tf
            self.tf = tf
            self._device = None  # TensorFlow handles device placement automatically
        else:  # pytorch
            import torch
            self.torch = torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate linearly spaced beta values from beta_start to beta_end
        betas = numpy.linspace(
            beta_start,
            beta_end,
            time_steps,
            dtype=numpy.float32
        )

        self._number_time_steps = int(time_steps)

        # Calculate alpha values (1 - beta)
        alphas = 1.0 - betas

        # Compute cumulative product of alphas (ᾱ_t = ∏_{s=1}^t α_s)
        alphas_cumulative_product = numpy.cumprod(alphas, axis=0)

        # Compute cumulative product of alphas shifted by one time step (ᾱ_{t-1})
        alphas_cumulative_product_previous = numpy.append(1.0, alphas_cumulative_product[:-1])

        # Convert arrays to framework-specific tensors
        self._betas = self._to_tensor(betas)
        self._alphas_cumulative_product = self._to_tensor(alphas_cumulative_product)
        self._alphas_cumulative_product_previous = self._to_tensor(alphas_cumulative_product_previous)
        self._sqrt_alphas_cumulative_product = self._to_tensor(numpy.sqrt(alphas_cumulative_product))
        self._sqrt_one_minus_alphas_cumulative_product = self._to_tensor(numpy.sqrt(1.0 - alphas_cumulative_product))
        self._log_one_minus_alphas_cumulative_product = self._to_tensor(numpy.log(1.0 - alphas_cumulative_product))
        self._sqrt_recip_alphas_cumulative_product = self._to_tensor(numpy.sqrt(1.0 / alphas_cumulative_product))
        self._sqrt_recipm1_alphas_cumulative_product = self._to_tensor(
            numpy.sqrt(1.0 / alphas_cumulative_product - 1))

        # Calculate posterior parameters
        _posterior_variance = (betas * (1.0 - alphas_cumulative_product_previous) / (1.0 - alphas_cumulative_product))
        self._posterior_variance = self._to_tensor(_posterior_variance)
        self._posterior_log_variance_clipped = self._to_tensor(numpy.log(numpy.maximum(_posterior_variance, 1e-20)))
        self._posterior_mean_first_coefficient = self._to_tensor(
            betas * numpy.sqrt(alphas_cumulative_product_previous) / (1.0 - alphas_cumulative_product))
        self._posterior_mean_second_coefficient = self._to_tensor(
            (1.0 - alphas_cumulative_product_previous) * numpy.sqrt(alphas) / (1.0 - alphas_cumulative_product))

    def _to_tensor(self, array):
        """
        Converts a numpy array to a framework-specific tensor.

        Parameters:
        -----------
            array : numpy.ndarray
                Array to convert.

        Returns:
        --------
            Tensor in the appropriate framework format.
        """
        if self._framework == 'tensorflow':
            return self.tf.constant(array, dtype=self.tf.float32)
        else:  # pytorch
            return self.torch.tensor(array, dtype=self.torch.float32, device=self._device)

    def _extract(self, a, t, x_shape):
        """
        Extracts values from a tensor based on the time index and reshapes them to match the input batch.

        Parameters:
        -----------
            a : Tensor
                Tensor from which values are extracted.
            t : Tensor
                Time step indices.
            x_shape : Tensor or tuple
                Shape of the input tensor.

        Returns:
        --------
            Tensor
                Extracted and reshaped values.
        """
        if self._framework == 'tensorflow':
            return self._extract_tensorflow(a, t, x_shape)
        else:
            return self._extract_pytorch(a, t, x_shape)

    def _extract_tensorflow(self, a, t, x_shape):
        """TensorFlow implementation of _extract."""
        batch_size = x_shape[0]
        out = self.tf.gather(a, t)
        return self.tf.reshape(out, [batch_size, 1, 1])

    def _extract_pytorch(self, a, t, x_shape):
        """PyTorch implementation of _extract."""
        batch_size = x_shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, 1, 1, 1)

    def q_mean_variance(self, x_start, t):
        """
        Computes the mean, variance, and log variance of the forward diffusion process at a given time step.

        Parameters:
        -----------
            x_start : Tensor
                Original input data.
            t : Tensor
                Time step indices.

        Returns:
        --------
            tuple of Tensor
                Mean, variance, and log variance of the forward process.
        """
        if self._framework == 'tensorflow':
            x_start_shape = self.tf.shape(x_start)
        else:
            x_start_shape = x_start.shape

        mean = self._extract(self._sqrt_alphas_cumulative_product, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self._alphas_cumulative_product, t, x_start_shape)
        log_variance = self._extract(self._log_one_minus_alphas_cumulative_product, t, x_start_shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """
        Samples a noisy version of the input at a given time step.

        Parameters:
        -----------
            x_start : Tensor
                Original input data.
            t : Tensor
                Time step indices.
            noise : Tensor
                Gaussian noise to add.

        Returns:
        --------
            Tensor
                Noisy sample at time step t.
        """
        if self._framework == 'tensorflow':
            x_start_shape = self.tf.shape(x_start)
        else:
            x_start_shape = x_start.shape

        return (self._extract(self._sqrt_alphas_cumulative_product, t, x_start_shape) * x_start
                + self._extract(self._sqrt_one_minus_alphas_cumulative_product, t, x_start_shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predicts the original input x_start given a noisy sample x_t and noise.

        Parameters:
        -----------
            x_t : Tensor
                Noisy input data at time step t.
            t : Tensor
                Time step indices.
            noise : Tensor
                Gaussian noise applied during diffusion.

        Returns:
        --------
            Tensor
                Predicted x_start.
        """
        if self._framework == 'tensorflow':
            x_t_shape = self.tf.shape(x_t)
        else:
            x_t_shape = x_t.shape

        return (self._extract(self._sqrt_recip_alphas_cumulative_product, t, x_t_shape) * x_t
                - self._extract(self._sqrt_recipm1_alphas_cumulative_product, t, x_t_shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        """
        Computes the posterior mean and variance for the reverse diffusion process.

        Parameters:
        -----------
            x_start : Tensor
                Original input data.
            x_t : Tensor
                Noisy input at time step t.
            t : Tensor
                Time step indices.

        Returns:
        --------
            tuple of Tensor
                Posterior mean, variance, and log variance.
        """
        if self._framework == 'tensorflow':
            x_t_shape = self.tf.shape(x_t)
        else:
            x_t_shape = x_t.shape

        posterior_mean = (self._extract(self._posterior_mean_first_coefficient, t, x_t_shape) * x_start
                          + self._extract(self._posterior_mean_second_coefficient, t, x_t_shape) * x_t)
        posterior_variance = self._extract(self._posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self._posterior_log_variance_clipped, t, x_t_shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, predicted_noise, x, t, clip_denoised=True):
        """
        Predicts the mean and variance of the model's posterior distribution at time step t.

        Parameters:
        -----------
            predicted_noise : Tensor
                Noise predicted by the model.
            x : Tensor
                Noisy input at time step t.
            t : Tensor
                Time step indices.
            clip_denoised : bool, optional
                Whether to clip the denoised output.

        Returns:
        --------
            tuple of Tensor
                Predicted mean, variance, and log variance.
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)

        if clip_denoised:
            if self._framework == 'tensorflow':
                x_recon = self.tf.clip_by_value(x_recon, self._clip_min, self._clip_max)
            else:
                x_recon = self.torch.clamp(x_recon, self._clip_min, self._clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, predicted_noise, x, t, clip_denoised=True):
        """
        Samples from the model's posterior distribution at a given time step.

        Parameters:
        -----------
            predicted_noise : Tensor
                Noise predicted by the model.
            x : Tensor
                Noisy input at time step t.
            t : Tensor
                Time step indices.
            clip_denoised : bool, optional
                Whether to clip the denoised output.

        Returns:
        --------
            Tensor
                Sampled output at time step t.
        """
        if self._framework == 'tensorflow':
            return self._p_sample_tensorflow(predicted_noise, x, t, clip_denoised)
        else:
            return self._p_sample_pytorch(predicted_noise, x, t, clip_denoised)

    def _p_sample_tensorflow(self, predicted_noise, x, t, clip_denoised):
        """TensorFlow implementation of p_sample."""
        model_mean, _, model_log_variance = self.p_mean_variance(predicted_noise, x=x, t=t, 
                                                                 clip_denoised=clip_denoised)

        noise = self.tf.random.normal(shape=x.shape, dtype=x.dtype)
        nonzero_mask = self.tf.reshape(
            1 - self.tf.cast(self.tf.equal(t, 0), self.tf.float32),
            [self.tf.shape(x)[0], 1, 1, 1]
        )

        return model_mean + nonzero_mask * self.tf.exp(0.5 * model_log_variance) * noise

    def _p_sample_pytorch(self, predicted_noise, x, t, clip_denoised):
        """PyTorch implementation of p_sample."""
        model_mean, _, model_log_variance = self.p_mean_variance(predicted_noise, x=x, t=t, 
                                                                 clip_denoised=clip_denoised)

        noise = self.torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return model_mean + nonzero_mask * self.torch.exp(0.5 * model_log_variance) * noise

    @property
    def framework(self) -> str:
        """Get the framework being used.

        Returns:
            The framework name ('tensorflow' or 'pytorch').
        """
        return self._framework

    @property
    def beta_start(self) -> float:
        """Get the starting value of beta for the noise schedule.

        Returns:
            The initial beta value for the diffusion process.
        """
        return self._beta_start

    @beta_start.setter
    def beta_start(self, value: float) -> None:
        """Set the starting value of beta for the noise schedule.

        Args:
            value: The initial beta value (must be positive and <= beta_end).

        Raises:
            ValueError: If value is not positive or is greater than beta_end.
        """
        if value <= 0:
            raise ValueError("beta_start must be positive")

        if hasattr(self, '_beta_end') and value > self._beta_end:
            raise ValueError("beta_start must be less than or equal to beta_end")

        self._beta_start = value

    @property
    def beta_end(self) -> float:
        """Get the ending value of beta for the noise schedule.

        Returns:
            The final beta value for the diffusion process.
        """
        return self._beta_end

    @beta_end.setter
    def beta_end(self, value: float) -> None:
        """Set the ending value of beta for the noise schedule.

        Args:
            value: The final beta value (must be positive and >= beta_start).

        Raises:
            ValueError: If value is not positive or is less than beta_start.
        """
        if value <= 0:
            raise ValueError("beta_end must be positive")

        if hasattr(self, '_beta_start') and value < self._beta_start:
            raise ValueError("beta_end must be greater than or equal to beta_start")

        self._beta_end = value

    @property
    def time_steps(self) -> int:
        """Get the number of diffusion time steps.

        Returns:
            The number of time steps in the diffusion process.
        """
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int) -> None:
        """Set the number of diffusion time steps.

        Args:
            value: The number of time steps (must be positive integer).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("time_steps must be a positive integer")

        self._time_steps = value

    @property
    def clip_min(self) -> float:
        """Get the minimum clipping value for the output.

        Returns:
            The minimum value to clip outputs to.
        """
        return self._clip_min

    @clip_min.setter
    def clip_min(self, value: float) -> None:
        """Set the minimum clipping value for the output.

        Args:
            value: The minimum clip value (must be < clip_max).

        Raises:
            ValueError: If value is not less than clip_max.
        """
        if hasattr(self, '_clip_max') and value >= self._clip_max:
            raise ValueError("clip_min must be less than clip_max")

        self._clip_min = value

    @property
    def clip_max(self) -> float:
        """Get the maximum clipping value for the output.

        Returns:
            The maximum value to clip outputs to.
        """
        return self._clip_max

    @clip_max.setter
    def clip_max(self, value: float) -> None:
        """Set the maximum clipping value for the output.

        Args:
            value: The maximum clip value (must be > clip_min).

        Raises:
            ValueError: If value is not greater than clip_min.
        """
        if hasattr(self, '_clip_min') and value <= self._clip_min:
            raise ValueError("clip_max must be greater than clip_min")

        self._clip_max = value

    @property
    def device(self):
        """Get the device being used (PyTorch only).

        Returns:
            Device object for PyTorch, None for TensorFlow.
        """
        return self._device
