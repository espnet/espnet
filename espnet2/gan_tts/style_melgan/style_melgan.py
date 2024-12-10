# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""StyleMelGAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import copy
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from espnet2.gan_tts.melgan import MelGANDiscriminator as BaseDiscriminator
from espnet2.gan_tts.melgan.pqmf import PQMF
from espnet2.gan_tts.style_melgan.tade_res_block import TADEResBlock


class StyleMelGANGenerator(torch.nn.Module):
    """
        Style MelGAN generator module.

    This module implements the StyleMelGAN generator, which is a neural network
    architecture designed for generating high-quality audio waveforms from
    mel-spectrograms and auxiliary noise inputs.

    Attributes:
        in_channels (int): Number of input noise channels.
        noise_upsample (torch.nn.Sequential): Sequential model for noise upsampling.
        blocks (torch.nn.ModuleList): List of TADEResBlock modules for audio
            generation.
        upsample_factor (int): Total upsampling factor for output generation.
        output_conv (torch.nn.Sequential): Final convolutional layer for output.

    Args:
        in_channels (int): Number of input noise channels.
        aux_channels (int): Number of auxiliary input channels.
        channels (int): Number of channels for convolutional layers.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of convolutional layers.
        dilation (int): Dilation factor for convolutional layers.
        bias (bool): Whether to add bias parameter in convolution layers.
        noise_upsample_scales (List[int]): List of noise upsampling scales.
        noise_upsample_activation (str): Activation function for noise upsampling.
        noise_upsample_activation_params (Dict[str, Any]): Hyperparameters for
            the activation function.
        upsample_scales (List[int]): List of upsampling scales.
        upsample_mode (str): Upsampling mode in TADE layer.
        gated_function (str): Gated function used in TADEResBlock
            ("softmax" or "sigmoid").
        use_weight_norm (bool): Whether to use weight normalization.

    Returns:
        None

    Examples:
        # Creating a StyleMelGAN generator instance
        generator = StyleMelGANGenerator(in_channels=128, aux_channels=80)

        # Forward pass with auxiliary input and noise
        c = torch.randn(1, 80, 100)  # Auxiliary input
        z = torch.randn(1, 128, 1)   # Noise input
        output = generator(c, z)

    Note:
        This module is modified from the original implementation found at
        https://github.com/kan-bayashi/ParallelWaveGAN.
    """

    def __init__(
        self,
        in_channels: int = 128,
        aux_channels: int = 80,
        channels: int = 64,
        out_channels: int = 1,
        kernel_size: int = 9,
        dilation: int = 2,
        bias: bool = True,
        noise_upsample_scales: List[int] = [11, 2, 2, 2],
        noise_upsample_activation: str = "LeakyReLU",
        noise_upsample_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        upsample_scales: List[int] = [2, 2, 2, 2, 2, 2, 2, 2, 1],
        upsample_mode: str = "nearest",
        gated_function: str = "softmax",
        use_weight_norm: bool = True,
    ):
        """Initilize StyleMelGANGenerator module.

        Args:
            in_channels (int): Number of input noise channels.
            aux_channels (int): Number of auxiliary input channels.
            channels (int): Number of channels for conv layer.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of conv layers.
            dilation (int): Dilation factor for conv layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            noise_upsample_scales (List[int]): List of noise upsampling scales.
            noise_upsample_activation (str): Activation function module name for noise
                upsampling.
            noise_upsample_activation_params (Dict[str, Any]): Hyperparameters for the
                above activation function.
            upsample_scales (List[int]): List of upsampling scales.
            upsample_mode (str): Upsampling mode in TADE layer.
            gated_function (str): Gated function used in TADEResBlock
                ("softmax" or "sigmoid").
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.in_channels = in_channels

        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            # NOTE(kan-bayashi): How should we design noise upsampling part?
            noise_upsample += [
                torch.nn.ConvTranspose1d(
                    in_chs,
                    channels,
                    noise_upsample_scale * 2,
                    stride=noise_upsample_scale,
                    padding=noise_upsample_scale // 2 + noise_upsample_scale % 2,
                    output_padding=noise_upsample_scale % 2,
                    bias=bias,
                )
            ]
            noise_upsample += [
                getattr(torch.nn, noise_upsample_activation)(
                    **noise_upsample_activation_params
                )
            ]
            in_chs = channels
        self.noise_upsample = torch.nn.Sequential(*noise_upsample)
        self.noise_upsample_factor = int(np.prod(noise_upsample_scales))

        self.blocks = torch.nn.ModuleList()
        aux_chs = aux_channels
        for upsample_scale in upsample_scales:
            self.blocks += [
                TADEResBlock(
                    in_channels=channels,
                    aux_channels=aux_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                    upsample_factor=upsample_scale,
                    upsample_mode=upsample_mode,
                    gated_function=gated_function,
                ),
            ]
            aux_chs = channels
        self.upsample_factor = int(np.prod(upsample_scales) * out_channels)

        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                out_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(
        self, c: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method computes the forward pass of the StyleMelGAN generator. It
        takes an auxiliary input tensor and an optional noise tensor, and produces
        an output tensor. If the noise tensor is not provided, a random tensor is
        generated.

        Args:
            c (Tensor): Auxiliary input tensor (B, channels, T).
            z (Optional[Tensor]): Input noise tensor (B, in_channels, 1). If
                not provided, a random tensor will be generated.

        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).

        Examples:
            >>> generator = StyleMelGANGenerator()
            >>> aux_input = torch.randn(4, 64, 100)  # Batch of 4, 64 channels, length 100
            >>> noise_input = torch.randn(4, 128, 1)  # Batch of 4, 128 noise channels
            >>> output = generator.forward(aux_input, noise_input)
            >>> output.shape
            torch.Size([4, 1, 2000])  # Example output shape

            >>> output_random_noise = generator.forward(aux_input)
            >>> output_random_noise.shape
            torch.Size([4, 1, 2000])  # Example output shape with random noise
        """
        if z is None:
            z = torch.randn(c.size(0), self.in_channels, 1).to(
                device=c.device,
                dtype=c.dtype,
            )
        x = self.noise_upsample(z)
        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)
        return x

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all layers.

        This method iterates through all layers of the StyleMelGANGenerator and
        removes the weight normalization applied to the convolutional layers. It
        utilizes the `torch.nn.utils.remove_weight_norm` function to perform the
        operation. If a layer does not have weight normalization, a ValueError
        is caught and ignored.

        Examples:
            >>> generator = StyleMelGANGenerator(use_weight_norm=True)
            >>> generator.remove_weight_norm()  # Removes weight normalization
        """

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.

        This method applies weight normalization to all convolutional layers in
        the generator. Weight normalization is a technique that can help to
        stabilize training and improve convergence.

        It specifically targets layers of type `torch.nn.Conv1d` and
        `torch.nn.ConvTranspose1d`. After applying weight normalization, the
        model's layers will be able to utilize the benefits of this
        normalization technique during training.

        Examples:
            >>> model = StyleMelGANGenerator()
            >>> model.apply_weight_norm()
            # This will apply weight normalization to all Conv1d layers in the model.

        Note:
            It is recommended to apply weight normalization during the
            initialization of the model to achieve optimal training performance.
        """

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """
                Reset parameters.

        This method resets the weights of the convolutional layers within the model
        to a normal distribution with a mean of 0.0 and a standard deviation of 0.02.
        This is typically used to initialize the weights of the model before training
        or after loading a pre-trained model to ensure the model starts with a fresh
        set of parameters.

        The function applies the reset operation to all instances of `Conv1d` and
        `ConvTranspose1d` layers in the model.

        Examples:
            To reset parameters of a `StyleMelGANGenerator` instance, you can call:

            ```python
            generator = StyleMelGANGenerator()
            generator.reset_parameters()
            ```

            Similarly, for a `StyleMelGANDiscriminator` instance:

            ```python
            discriminator = StyleMelGANDiscriminator()
            discriminator.reset_parameters()
            ```

        Note:
            This function is automatically called during the initialization of the
            generator and discriminator classes.
        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c: torch.Tensor) -> torch.Tensor:
        """
        Perform inference.

        This method takes an input tensor and generates an output tensor
        using the trained StyleMelGAN generator. The input tensor is
        expected to be of shape (T, in_channels), where T is the time
        dimension and in_channels corresponds to the number of input
        channels.

        Args:
            c (Tensor): Input tensor of shape (T, in_channels).

        Returns:
            Tensor: Output tensor of shape (T ** prod(upsample_scales),
            out_channels).

        Examples:
            >>> generator = StyleMelGANGenerator()
            >>> input_tensor = torch.randn(100, 128)  # Example input
            >>> output_tensor = generator.inference(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([1000, 1])  # Output shape based on upsampling

        Note:
            The input tensor is transposed and reshaped before processing.
            Additionally, noise is generated internally to aid in the
            inference process.
        """
        c = c.transpose(1, 0).unsqueeze(0)

        # prepare noise input
        noise_size = (
            1,
            self.in_channels,
            math.ceil(c.size(2) / self.noise_upsample_factor),
        )
        noise = torch.randn(*noise_size, dtype=torch.float).to(
            next(self.parameters()).device
        )
        x = self.noise_upsample(noise)

        # NOTE(kan-bayashi): To remove pop noise at the end of audio, perform padding
        #    for feature sequence and after generation cut the generated audio. This
        #    requires additional computation but it can prevent pop noise.
        total_length = c.size(2) * self.upsample_factor
        c = F.pad(c, (0, x.size(2) - c.size(2)), "replicate")

        # This version causes pop noise.
        # x = x[:, :, :c.size(2)]

        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)[..., :total_length]

        return x.squeeze(0).transpose(1, 0)


class StyleMelGANDiscriminator(torch.nn.Module):
    """
    Style MelGAN discriminator module.

    This module serves as a discriminator for the Style MelGAN architecture,
    which is designed to evaluate the quality of generated audio signals. It
    uses a combination of PQMF (Polyphase Quadrature Mirror Filter) and a base
    discriminator to assess the authenticity of audio samples.

    Attributes:
        repeats (int): Number of repetitions to apply Random Window Discrimination (RWD).
        window_sizes (List[int]): List of random window sizes for analysis.
        pqmfs (ModuleList): List of PQMF modules for downsampling.
        discriminators (ModuleList): List of base discriminators.

    Args:
        repeats (int): Number of repetitions to apply RWD.
        window_sizes (List[int]): List of random window sizes.
        pqmf_params (List[List[int]]): Parameters for PQMF modules.
        discriminator_params (Dict[str, Any]): Parameters for the base discriminator.
        use_weight_norm (bool): Whether to apply weight normalization.

    Raises:
        AssertionError: If the lengths of window_sizes and pqmf_params do not match
            or if the sum of calculated sizes does not match the length of
            window_sizes.

    Examples:
        >>> discriminator = StyleMelGANDiscriminator()
        >>> input_tensor = torch.randn(8, 1, 2048)  # Batch of 8 samples
        >>> outputs = discriminator(input_tensor)
        >>> print(len(outputs))  # Output will be the number of discriminators * repeats

    Note:
        The `pqmf_params` should be defined carefully to match the window sizes.
    """

    def __init__(
        self,
        repeats: int = 2,
        window_sizes: List[int] = [512, 1024, 2048, 4096],
        pqmf_params: List[List[int]] = [
            [1, None, None, None],
            [2, 62, 0.26700, 9.0],
            [4, 62, 0.14200, 9.0],
            [8, 62, 0.07949, 9.0],
        ],
        discriminator_params: Dict[str, Any] = {
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 16,
            "max_downsample_channels": 512,
            "bias": True,
            "downsample_scales": [4, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
            "pad": "ReflectionPad1d",
            "pad_params": {},
        },
        use_weight_norm: bool = True,
    ):
        """Initilize StyleMelGANDiscriminator module.

        Args:
            repeats (int): Number of repititons to apply RWD.
            window_sizes (List[int]): List of random window sizes.
            pqmf_params (List[List[int]]): List of list of Parameters for PQMF modules
            discriminator_params (Dict[str, Any]): Parameters for base discriminator
                module.
            use_weight_nom (bool): Whether to apply weight normalization.

        """
        super().__init__()

        # window size check
        assert len(window_sizes) == len(pqmf_params)
        sizes = [ws // p[0] for ws, p in zip(window_sizes, pqmf_params)]
        assert len(window_sizes) == sum([sizes[0] == size for size in sizes])

        self.repeats = repeats
        self.window_sizes = window_sizes
        self.pqmfs = torch.nn.ModuleList()
        self.discriminators = torch.nn.ModuleList()
        for pqmf_param in pqmf_params:
            d_params = copy.deepcopy(discriminator_params)
            d_params["in_channels"] = pqmf_param[0]
            if pqmf_param[0] == 1:
                self.pqmfs += [torch.nn.Identity()]
            else:
                self.pqmfs += [PQMF(*pqmf_param)]
            self.discriminators += [BaseDiscriminator(**d_params)]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate forward propagation.

        This method processes the input tensor through multiple discriminators and
        returns their outputs. It applies random windowing to the input for each
        discriminator to evaluate the audio features.

        Args:
            x (Tensor): Input tensor of shape (B, 1, T), where B is the batch
                size, and T is the length of the audio signal.

        Returns:
            List: A list of discriminator outputs. The number of items in the
                list will be equal to `repeats * number of discriminators`.

        Examples:
            >>> discriminator = StyleMelGANDiscriminator()
            >>> input_tensor = torch.randn(8, 1, 4096)  # Batch of 8 samples
            >>> outputs = discriminator(input_tensor)
            >>> print(len(outputs))  # Should equal repeats * number of discriminators
        """
        outs = []
        for _ in range(self.repeats):
            outs += self._forward(x)

        return outs

    def _forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for idx, (ws, pqmf, disc) in enumerate(
            zip(self.window_sizes, self.pqmfs, self.discriminators)
        ):
            # NOTE(kan-bayashi): Is it ok to apply different window for real and fake
            #   samples?
            start_idx = np.random.randint(x.size(-1) - ws)
            x_ = x[:, :, start_idx : start_idx + ws]
            if idx == 0:
                x_ = pqmf(x_)
            else:
                x_ = pqmf.analysis(x_)
            outs += [disc(x_)]
        return outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.

        This method applies weight normalization to all convolutional layers
        (both 1D convolution and transposed convolution) in the network. Weight
        normalization can help in stabilizing the training process and can lead
        to better convergence.

        It iterates through all the modules in the network and applies weight
        normalization if the module is an instance of `torch.nn.Conv1d` or
        `torch.nn.ConvTranspose1d`.

        Note:
            This function is typically called during the initialization of the
            model if the `use_weight_norm` flag is set to True.

        Examples:
            >>> model = StyleMelGANDiscriminator(use_weight_norm=True)
            >>> model.apply_weight_norm()  # Applies weight normalization to all layers
        """

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """
        Reset parameters of the discriminator's convolutional layers.

        This method iterates through all the modules in the discriminator and
        resets the weights of the convolutional layers (both `Conv1d` and
        `ConvTranspose1d`) to follow a normal distribution with mean 0.0 and
        standard deviation 0.02. This is commonly used to initialize weights
        before training a neural network to ensure better convergence.

        It also logs a debug message indicating that the parameters have been
        reset for each convolutional layer.

        Examples:
            >>> discriminator = StyleMelGANDiscriminator()
            >>> discriminator.reset_parameters()

        Note:
            This method is typically called during the initialization of the
            model to ensure that the parameters are set correctly before
            training begins.
        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
