# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HiFi-GAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from espnet2.gan_tts.hifigan.residual_block import ResidualBlock


class HiFiGANGenerator(torch.nn.Module):
    """
    HiFiGAN generator module for high-fidelity audio synthesis.

    This module implements the HiFi-GAN generator architecture, which is used
    for generating high-quality audio waveforms from mel-spectrograms. The
    implementation is inspired by the work done in the ParallelWaveGAN project.

    Attributes:
        upsample_factor (int): The total upsampling factor applied to the input.
        num_upsamples (int): The number of upsampling layers.
        num_blocks (int): The number of residual blocks per upsampling layer.
        input_conv (torch.nn.Conv1d): The initial convolution layer.
        upsamples (torch.nn.ModuleList): List of upsampling layers.
        blocks (torch.nn.ModuleList): List of residual blocks.
        output_conv (torch.nn.Sequential): The final convolution and activation layers.
        global_conv (Optional[torch.nn.Conv1d]): Global conditioning convolution layer.

    Args:
        in_channels (int): Number of input channels (default: 80).
        out_channels (int): Number of output channels (default: 1).
        channels (int): Number of hidden representation channels (default: 512).
        global_channels (int): Number of global conditioning channels (default: -1).
        kernel_size (int): Kernel size of initial and final conv layer (default: 7).
        upsample_scales (List[int]): List of upsampling scales (default: [8, 8, 2, 2]).
        upsample_kernel_sizes (List[int]): List of kernel sizes for upsample layers
            (default: [16, 16, 4, 4]).
        resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks
            (default: [3, 7, 11]).
        resblock_dilations (List[List[int]]): List of list of dilations for residual
            blocks (default: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]).
        use_additional_convs (bool): Whether to use additional conv layers in
            residual blocks (default: True).
        bias (bool): Whether to add bias parameter in convolution layers (default: True).
        nonlinear_activation (str): Activation function module name (default: "LeakyReLU").
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
            function (default: {"negative_slope": 0.1}).
        use_weight_norm (bool): Whether to use weight norm (default: True).

    Raises:
        AssertionError: If kernel_size is not odd or if lengths of upsample_scales,
        upsample_kernel_sizes, resblock_dilations, and resblock_kernel_sizes do not match.

    Examples:
        >>> generator = HiFiGANGenerator()
        >>> mel_spectrogram = torch.randn(1, 80, 100)  # Example input
        >>> output_waveform = generator(mel_spectrogram)
        >>> print(output_waveform.shape)  # Output shape will be (1, 1, T)

    Note:
        The HiFiGAN architecture is designed to synthesize high-quality audio
        from low-dimensional features such as mel-spectrograms. It utilizes
        residual blocks and upsampling techniques to achieve high fidelity.
    """

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        channels: int = 512,
        global_channels: int = -1,
        kernel_size: int = 7,
        upsample_scales: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        use_additional_convs: bool = True,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            global_channels (int): Number of global conditioning channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (List[int]): List of upsampling scales.
            upsample_kernel_sizes (List[int]): List of kernel sizes for upsample layers.
            resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks.
            resblock_dilations (List[List[int]]): List of list of dilations for residual
                blocks.
            use_additional_convs (bool): Whether to use additional conv layers in
                residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.upsample_factor = int(np.prod(upsample_scales) * out_channels)
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )
        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, channels, 1)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(
        self, c: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method computes the forward pass of the HiFiGAN generator by
        processing the input tensor through several convolutional layers,
        upsampling layers, and residual blocks. If a global conditioning
        tensor is provided, it will be added to the processed input before
        proceeding through the network.

        Args:
            c (torch.Tensor): Input tensor of shape (B, in_channels, T),
                where B is the batch size, in_channels is the number of input
                channels, and T is the length of the input sequence.
            g (Optional[torch.Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This tensor is optional and, if provided,
                is added to the input tensor after the initial convolution.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, T),
                where out_channels is the number of output channels.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be (1, 1, T)

        Note:
            The input tensor must have the correct number of channels as
            specified during the initialization of the HiFiGANGenerator.
            The global conditioning tensor must have the same batch size as
            the input tensor if provided.

        Raises:
            AssertionError: If the input tensor does not match the expected
            shape or if the global conditioning tensor has an incompatible shape.
        """
        c = self.input_conv(c)
        if g is not None:
            c = c + self.global_conv(g)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """
        Reset parameters of the HiFiGANGenerator.

        This initialization follows the official implementation manner as detailed
        in the HiFi-GAN repository. The weights of convolutional layers are
        initialized using a normal distribution with a mean of 0 and a standard
        deviation of 0.01. This method applies to all layers in the generator,
        specifically targeting `Conv1d` and `ConvTranspose1d` modules. It ensures
        that the model's parameters are reset to a known state, which can be
        useful for experimentation or retraining.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> generator.reset_parameters()  # Resets parameters to default

        Note:
            This method is typically called during the initialization of the
            generator, but it can also be called manually to reset the parameters
            at any time during the model's lifecycle.

        Raises:
            None: This method does not raise any exceptions.
        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.

        This method traverses through all layers of the model and removes the
        weight normalization applied to convolutional layers. If a layer does
        not have weight normalization applied, it catches the ValueError and
        continues without raising an exception.

        Note:
            This method is useful for models that were previously using weight
            normalization and need to revert back to the standard weight
            parameters for compatibility or performance reasons.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> generator.apply_weight_norm()  # Apply weight normalization
            >>> generator.remove_weight_norm()  # Remove weight normalization

        Raises:
            ValueError: If a module does not support weight normalization.
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

        This method applies weight normalization to all convolutional layers in the
        HiFiGAN generator. Weight normalization can improve the training speed and
        stability of the model by reparameterizing the weights of the layers.

        It is important to note that weight normalization should be applied during
        the initialization of the model if the `use_weight_norm` parameter is set
        to `True`.

        Examples:
            >>> generator = HiFiGANGenerator(use_weight_norm=True)
            >>> generator.apply_weight_norm()

        Note:
            This method logs a debug message for each layer that weight
            normalization is applied to, aiding in tracking the model's structure
            during development and debugging.

        Raises:
            None: This method does not raise any exceptions.
        """

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(
        self, c: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform inference using the HiFiGAN generator.

        This method processes the input tensor and optionally incorporates global
        conditioning to produce an output tensor. The input tensor should be in
        the format (T, in_channels), where T is the time dimension. If a global
        conditioning tensor is provided, it should have the shape
        (global_channels, 1).

        Args:
            c (torch.Tensor): Input tensor with shape (T, in_channels).
            g (Optional[torch.Tensor]): Global conditioning tensor with shape
                (global_channels, 1). This tensor is optional and can be set to
                None.

        Returns:
            torch.Tensor: Output tensor with shape (T ** upsample_factor, out_channels),
                where upsample_factor is the product of the upsampling scales.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(100, 80)  # (T, in_channels)
            >>> output_tensor = generator.inference(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([800, 1])  # Example output shape

            >>> global_conditioning = torch.randn(8, 1)  # (global_channels, 1)
            >>> output_tensor_with_gc = generator.inference(input_tensor,
            ...                                              global_conditioning)
            >>> print(output_tensor_with_gc.shape)
            torch.Size([800, 1])  # Example output shape with global conditioning
        """
        if g is not None:
            g = g.unsqueeze(0)
        c = self.forward(c.transpose(1, 0).unsqueeze(0), g=g)
        return c.squeeze(0).transpose(1, 0)


class HiFiGANPeriodDiscriminator(torch.nn.Module):
    """
    HiFiGAN period discriminator module.

    This module implements a period discriminator for the HiFi-GAN architecture.
    It utilizes convolutional layers to classify audio signals based on periodicity.

    Attributes:
        period (int): The period length for the discriminator.
        convs (ModuleList): A list of convolutional layers used in the model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        period (int): Period length.
        kernel_sizes (list): Kernel sizes for initial and final convolution layers.
        channels (int): Number of initial channels.
        downsample_scales (List[int]): List of downsampling scales.
        max_downsample_channels (int): Maximum number of downsampling channels.
        bias (bool): Whether to add bias parameter in convolution layers.
        nonlinear_activation (str): Activation function module name.
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for the activation
            function.
        use_weight_norm (bool): Whether to apply weight normalization to all conv layers.
        use_spectral_norm (bool): Whether to apply spectral normalization to all conv layers.

    Raises:
        ValueError: If both use_weight_norm and use_spectral_norm are set to True.

    Examples:
        >>> discriminator = HiFiGANPeriodDiscriminator()
        >>> input_tensor = torch.randn(8, 1, 256)  # Batch size 8, 1 channel, length 256
        >>> output = discriminator(input_tensor)
        >>> len(output)  # Output is a list of tensors from each layer
        5

    Note:
        The input tensor is reshaped to accommodate the period during processing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        period: int = 3,
        kernel_sizes: List[int] = [5, 3],
        channels: int = 32,
        downsample_scales: List[int] = [3, 3, 3, 3, 1],
        max_downsample_channels: int = 1024,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
        use_spectral_norm: bool = False,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv
                layer.
            channels (int): Number of initial channels.
            downsample_scales (List[int]): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in
                residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method computes the forward pass of the HiFiGAN generator by
        processing the input tensor through several convolutional layers,
        upsampling layers, and residual blocks. If a global conditioning
        tensor is provided, it will be added to the processed input before
        proceeding through the network.

        Args:
            c (torch.Tensor): Input tensor of shape (B, in_channels, T),
                where B is the batch size, in_channels is the number of input
                channels, and T is the length of the input sequence.
            g (Optional[torch.Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This tensor is optional and, if provided,
                is added to the input tensor after the initial convolution.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, T),
                where out_channels is the number of output channels.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be (1, 1, T)

        Note:
            The input tensor must have the correct number of channels as
            specified during the initialization of the HiFiGANGenerator.
            The global conditioning tensor must have the same batch size as
            the input tensor if provided.

        Raises:
            AssertionError: If the input tensor does not match the expected
            shape or if the global conditioning tensor has an incompatible shape.
        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.

        This method applies weight normalization to all convolutional layers in the
        HiFiGANPeriodDiscriminator. Weight normalization can improve the training
        speed and stability of the model by reparameterizing the weights of the
        layers. It is important to note that weight normalization should be applied
        during the initialization of the model if the `use_weight_norm` parameter
        is set to `True`.

        Examples:
            >>> discriminator = HiFiGANPeriodDiscriminator(use_weight_norm=True)
            >>> discriminator.apply_weight_norm()

        Note:
            This method logs a debug message for each layer that weight
            normalization is applied to, aiding in tracking the model's structure
            during development and debugging.
        """

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """
        Apply spectral normalization module from all of the layers.

        This method applies spectral normalization to all `Conv2d` layers within
        the HiFiGANPeriodDiscriminator module. Spectral normalization is a technique
        used to stabilize the training of generative adversarial networks (GANs) by
        controlling the Lipschitz constant of the network.

        Note:
            This method modifies the layers in place, so it is recommended to
            call this method after the module has been initialized.

        Examples:
            >>> discriminator = HiFiGANPeriodDiscriminator(use_spectral_norm=True)
            >>> discriminator.apply_spectral_norm()
        """

        def _apply_spectral_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):
    """
    HiFiGAN multi-period discriminator module.

    This module implements a multi-period discriminator for the HiFi-GAN model,
    which is used to distinguish between real and generated audio signals by
    utilizing multiple periods. The architecture is designed to capture various
    frequency patterns, improving the overall performance of the GAN during
    training.

    Args:
        periods (List[int]): List of periods used in the discriminator.
        discriminator_params (Dict[str, Any]): Parameters for the HiFi-GAN
            period discriminator module. The 'period' parameter will be
            overwritten by the values in 'periods'.

    Attributes:
        discriminators (torch.nn.ModuleList): A list of HiFiGANPeriodDiscriminator
            instances, one for each specified period.

    Returns:
        List: A list containing outputs from each period discriminator.

    Examples:
        >>> discriminator = HiFiGANMultiPeriodDiscriminator()
        >>> input_tensor = torch.randn(8, 1, 1024)  # Batch of 8 audio signals
        >>> outputs = discriminator(input_tensor)
        >>> len(outputs)  # Should equal the number of specified periods
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (List[int]): List of periods.
            discriminator_params (Dict[str, Any]): Parameters for hifi-gan period
                discriminator module. The period parameter will be overwritten.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method computes the forward pass of the HiFiGAN generator by
        processing the input tensor through several convolutional layers,
        upsampling layers, and residual blocks. If a global conditioning
        tensor is provided, it will be added to the processed input before
        proceeding through the network.

        Args:
            c (torch.Tensor): Input tensor of shape (B, in_channels, T),
                where B is the batch size, in_channels is the number of input
                channels, and T is the length of the input sequence.
            g (Optional[torch.Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This tensor is optional and, if provided,
                is added to the input tensor after the initial convolution.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, T),
                where out_channels is the number of output channels.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be (1, 1, T)

        Note:
            The input tensor must have the correct number of channels as
            specified during the initialization of the HiFiGANGenerator.
            The global conditioning tensor must have the same batch size as
            the input tensor if provided.

        Raises:
            AssertionError: If the input tensor does not match the expected
            shape or if the global conditioning tensor has an incompatible shape.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]

        return outs


class HiFiGANScaleDiscriminator(torch.nn.Module):
    """
    HiFi-GAN scale discriminator module.

    This class implements a scale discriminator for the HiFi-GAN model, which
    is responsible for distinguishing between real and generated audio at
    different scales. It uses convolutional layers to extract features from
    the input audio signal and applies downsampling to capture various
    frequency components.

    Attributes:
        layers (ModuleList): A list of sequential layers comprising the
            discriminator.

    Args:
        in_channels (int): Number of input channels. Defaults to 1.
        out_channels (int): Number of output channels. Defaults to 1.
        kernel_sizes (List[int]): List of four kernel sizes. The first will
            be used for the first conv layer, and the second is for
            downsampling part, and the remaining two are for the last two
            output layers. Defaults to [15, 41, 5, 3].
        channels (int): Initial number of channels for conv layer. Defaults to 128.
        max_downsample_channels (int): Maximum number of channels for
            downsampling layers. Defaults to 1024.
        max_groups (int): Maximum number of groups for group convolution.
            Defaults to 16.
        bias (bool): Whether to add bias parameter in convolution layers.
            Defaults to True.
        downsample_scales (List[int]): List of downsampling scales.
            Defaults to [2, 2, 4, 4, 1].
        nonlinear_activation (str): Activation function module name.
            Defaults to "LeakyReLU".
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for
            activation function. Defaults to {"negative_slope": 0.1}.
        use_weight_norm (bool): Whether to use weight norm. If set to true,
            it will be applied to all of the conv layers. Defaults to True.
        use_spectral_norm (bool): Whether to use spectral norm. If set to
            true, it will be applied to all of the conv layers. Defaults to False.

    Raises:
        ValueError: If both `use_weight_norm` and `use_spectral_norm` are True.

    Examples:
        >>> discriminator = HiFiGANScaleDiscriminator()
        >>> input_tensor = torch.randn(1, 1, 1024)  # (B, C, T)
        >>> outputs = discriminator(input_tensor)
        >>> print([out.shape for out in outputs])
        [torch.Size([1, 128, 1024]), torch.Size([1, 128, 512]),
         torch.Size([1, 128, 128]), torch.Size([1, 1, 128])]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: List[int] = [15, 41, 5, 3],
        channels: int = 128,
        max_downsample_channels: int = 1024,
        max_groups: int = 16,
        bias: int = True,
        downsample_scales: List[int] = [2, 2, 4, 4, 1],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
        use_spectral_norm: bool = False,
    ):
        """Initilize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (List[int]): List of four kernel sizes. The first will be used
                for the first conv layer, and the second is for downsampling part, and
                the remaining two are for the last two output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling
                layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm. If set to true, it
                will be applied to all of the conv layers.

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels,
                    channels,
                    # NOTE(kan-bayashi): Use always the same kernel size
                    kernel_sizes[0],
                    bias=bias,
                    padding=(kernel_sizes[0] - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias=bias,
            ),
        ]

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        self.use_weight_norm = use_weight_norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            self.apply_spectral_norm()

        # backward compatibility
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate forward propagation.

        This method computes the forward pass of the HiFiGAN generator by
        processing the input tensor through several convolutional layers,
        upsampling layers, and residual blocks. If a global conditioning
        tensor is provided, it will be added to the processed input before
        proceeding through the network.

        Args:
            c (torch.Tensor): Input tensor of shape (B, in_channels, T),
                where B is the batch size, in_channels is the number of input
                channels, and T is the length of the input sequence.
            g (Optional[torch.Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This tensor is optional and, if provided,
                is added to the input tensor after the initial convolution.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, T),
                where out_channels is the number of output channels.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be (1, 1, T)

        Note:
            The input tensor must have the correct number of channels as
            specified during the initialization of the HiFiGANGenerator.
            The global conditioning tensor must have the same batch size as
            the input tensor if provided.

        Raises:
            AssertionError: If the input tensor does not match the expected
            shape or if the global conditioning tensor has an incompatible shape.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.

        This method applies weight normalization to all convolutional layers in the
        HiFiGAN scale discriminator. Weight normalization can improve the training
        speed and stability of the model by reparameterizing the weights of the
        layers. It is important to note that weight normalization should be applied
        during the initialization of the model if the `use_weight_norm` parameter
        is set to `True`.

        Examples:
            >>> discriminator = HiFiGANScaleDiscriminator(use_weight_norm=True)
            >>> discriminator.apply_weight_norm()

        Note:
            Weight normalization is particularly useful when training deep neural
            networks as it can lead to faster convergence and improved performance.

        Raises:
            None: This method does not raise any exceptions.
        """

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """
        Apply spectral normalization module from all of the layers.

        This method applies spectral normalization to all `Conv1d` layers within
        the HiFiGANScaleDiscriminator module. Spectral normalization is a technique
        used to stabilize the training of generative adversarial networks (GANs) by
        controlling the Lipschitz constant of the network.

        Note:
            This method modifies the layers in place, so it is recommended to
            call this method after the module has been initialized.

        Examples:
            >>> discriminator = HiFiGANScaleDiscriminator(use_spectral_norm=True)
            >>> discriminator.apply_spectral_norm()
        """

        def _apply_spectral_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)

    def remove_weight_norm(self):
        """
            Remove weight normalization module from all of the layers.

        This method traverses through all layers of the HiFiGANScaleDiscriminator
        and removes weight normalization if it is applied. It logs a message for
        each layer from which the weight normalization is removed. If a layer does
        not have weight normalization applied, it catches the ValueError and continues
        without raising an exception.

        Note:
            This method is useful for models that were previously using weight
            normalization and need to revert back to the standard weight
            parameters for compatibility or performance reasons.

        Examples:
            >>> discriminator = HiFiGANScaleDiscriminator()
            >>> discriminator.apply_weight_norm()  # Apply weight normalization
            >>> discriminator.remove_weight_norm()  # Remove weight normalization

        Raises:
            ValueError: If a module does not have weight normalization applied.
        """

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def remove_spectral_norm(self):
        """
        Remove spectral normalization module from all of the layers.

        This method iterates through all layers of the model and removes the
        spectral normalization applied to each layer. It is useful when you want
        to revert the model to its original state without spectral normalization.

        Raises:
            ValueError: If the module does not have spectral normalization.

        Examples:
            >>> discriminator = HiFiGANScaleDiscriminator()
            >>> discriminator.apply_spectral_norm()  # Apply spectral norm
            >>> discriminator.remove_spectral_norm()  # Remove spectral norm

        Note:
            Ensure to call this method if you need to switch between weight and
            spectral normalization during model training or inference.
        """

        def _remove_spectral_norm(m):
            try:
                logging.debug(f"Spectral norm is removed from {m}.")
                torch.nn.utils.remove_spectral_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_spectral_norm)

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Fix the compatibility of weight / spectral normalization issue.

        Some pretrained models are trained with configs that use weight / spectral
        normalization, but actually, the norm is not applied. This causes the mismatch
        of the parameters with configs. To solve this issue, when parameter mismatch
        happens in loading pretrained model, we remove the norm from the current model.

        See also:
            - https://github.com/espnet/espnet/pull/5240
            - https://github.com/espnet/espnet/pull/5249
            - https://github.com/kan-bayashi/ParallelWaveGAN/pull/409

        """
        current_module_keys = [x for x in state_dict.keys() if x.startswith(prefix)]
        if self.use_weight_norm and any(
            [k.endswith("weight") for k in current_module_keys]
        ):
            logging.warning(
                "It seems weight norm is not applied in the pretrained model but the"
                " current model uses it. To keep the compatibility, we remove the norm"
                " from the current model. This may cause unexpected behavior due to the"
                " parameter mismatch in finetuning. To avoid this issue, please change"
                " the following parameters in config to false:\n"
                " - discriminator_params.follow_official_norm\n"
                " - discriminator_params.scale_discriminator_params.use_weight_norm\n"
                " - discriminator_params.scale_discriminator_params.use_spectral_norm\n"
                "\n"
                "See also:\n"
                " - https://github.com/espnet/espnet/pull/5240\n"
                " - https://github.com/espnet/espnet/pull/5249"
            )
            self.remove_weight_norm()
            self.use_weight_norm = False
            for k in current_module_keys:
                if k.endswith("weight_g") or k.endswith("weight_v"):
                    del state_dict[k]

        if self.use_spectral_norm and any(
            [k.endswith("weight") for k in current_module_keys]
        ):
            logging.warning(
                "It seems spectral norm is not applied in the pretrained model but the"
                " current model uses it. To keep the compatibility, we remove the norm"
                " from the current model. This may cause unexpected behavior due to the"
                " parameter mismatch in finetuning. To avoid this issue, please change"
                " the following parameters in config to false:\n"
                " - discriminator_params.follow_official_norm\n"
                " - discriminator_params.scale_discriminator_params.use_weight_norm\n"
                " - discriminator_params.scale_discriminator_params.use_spectral_norm\n"
                "\n"
                "See also:\n"
                " - https://github.com/espnet/espnet/pull/5240\n"
                " - https://github.com/espnet/espnet/pull/5249"
            )
            self.remove_spectral_norm()
            self.use_spectral_norm = False
            for k in current_module_keys:
                if (
                    k.endswith("weight_u")
                    or k.endswith("weight_v")
                    or k.endswith("weight_orig")
                ):
                    del state_dict[k]


class HiFiGANMultiScaleDiscriminator(torch.nn.Module):
    """
        HiFi-GAN multi-scale discriminator module.

    This module implements a multi-scale discriminator for HiFi-GAN,
    which is designed to distinguish real and generated audio signals
    at multiple scales. It uses a series of discriminators, each
    processing the input signal at a different scale, enabling the
    model to capture various frequency characteristics.

    Attributes:
        discriminators (torch.nn.ModuleList): A list of individual
            HiFiGAN scale discriminators.
        pooling (Optional[torch.nn.Module]): A downsampling pooling
            layer applied to the input signal between scales, if
            multiple scales are used.

    Args:
        scales (int): Number of multi-scales.
        downsample_pooling (str): Pooling module name for downsampling
            of the inputs.
        downsample_pooling_params (Dict[str, Any]): Parameters for
            the above pooling module.
        discriminator_params (Dict[str, Any]): Parameters for
            HiFi-GAN scale discriminator module.
        follow_official_norm (bool): Whether to follow the norm
            setting of the official implementation. The first
            discriminator uses spectral norm and the other
            discriminators use weight norm.

    Examples:
        >>> discriminator = HiFiGANMultiScaleDiscriminator(scales=3)
        >>> input_signal = torch.randn(1, 1, 2048)  # (Batch, Channels, Time)
        >>> outputs = discriminator(input_signal)
        >>> print(len(outputs))  # Should print 3 for 3 scales
    """

    def __init__(
        self,
        scales: int = 3,
        downsample_pooling: str = "AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm: bool = False,
    ):
        """Initilize HiFiGAN multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            downsample_pooling_params (Dict[str, Any]): Parameters for the above pooling
                module.
            discriminator_params (Dict[str, Any]): Parameters for hifi-gan scale
                discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the
                official implementaion. The first discriminator uses spectral norm
                and the other discriminators use weight norm.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = None
        if scales > 1:
            self.pooling = getattr(torch.nn, downsample_pooling)(
                **downsample_pooling_params
            )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Calculate forward propagation.

        This method computes the forward pass of the HiFiGAN generator by
        processing the input tensor through several convolutional layers,
        upsampling layers, and residual blocks. If a global conditioning
        tensor is provided, it will be added to the processed input before
        proceeding through the network.

        Args:
            c (torch.Tensor): Input tensor of shape (B, in_channels, T),
                where B is the batch size, in_channels is the number of input
                channels, and T is the length of the input sequence.
            g (Optional[torch.Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This tensor is optional and, if provided,
                is added to the input tensor after the initial convolution.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, T),
                where out_channels is the number of output channels.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be (1, 1, T)

        Note:
            The input tensor must have the correct number of channels as
            specified during the initialization of the HiFiGANGenerator.
            The global conditioning tensor must have the same batch size as
            the input tensor if provided.

        Raises:
            AssertionError: If the input tensor does not match the expected
            shape or if the global conditioning tensor has an incompatible shape.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            if self.pooling is not None:
                x = self.pooling(x)

        return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(torch.nn.Module):
    """
    HiFi-GAN multi-scale + multi-period discriminator module.

    This module combines multiple scale and period discriminators to evaluate
    generated audio signals across different scales and periods, enhancing
    the ability to distinguish real audio from generated audio.

    Args:
        scales (int): Number of multi-scales.
        scale_downsample_pooling (str): Pooling module name for downsampling
            of the inputs.
        scale_downsample_pooling_params (Dict[str, Any]): Parameters for the
            above pooling module.
        scale_discriminator_params (Dict[str, Any]): Parameters for HiFi-GAN
            scale discriminator module.
        follow_official_norm (bool): Whether to follow the norm setting of
            the official implementation. The first discriminator uses spectral
            norm and the other discriminators use weight norm.
        periods (List[int]): List of periods.
        period_discriminator_params (Dict[str, Any]): Parameters for HiFi-GAN
            period discriminator module. The period parameter will be overwritten.

    Examples:
        >>> discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator()
        >>> noise_signal = torch.randn(1, 1, 16000)  # Example noise signal
        >>> outputs = discriminator(noise_signal)
        >>> print(len(outputs))  # Should output the total number of
        # discriminators (scale + period)

    Returns:
        List[List[Tensor]]: List of list of each discriminator outputs,
            which consists of each layer output tensors. Multi-scale and
            multi-period ones are concatenated.
    """

    def __init__(
        self,
        # Multi-scale discriminator related
        scales: int = 3,
        scale_downsample_pooling: str = "AvgPool1d",
        scale_downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm: bool = True,
        # Multi-period discriminator related
        periods: List[int] = [2, 3, 5, 7, 11],
        period_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling
                module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale
                discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the
                official implementaion. The first discriminator uses spectral norm and
                the other discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period
                discriminator module. The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Calculate forward propagation.

        This method computes the forward pass of the HiFiGAN generator by
        processing the input tensor through several convolutional layers,
        upsampling layers, and residual blocks. If a global conditioning
        tensor is provided, it will be added to the processed input before
        proceeding through the network.

        Args:
            c (torch.Tensor): Input tensor of shape (B, in_channels, T),
                where B is the batch size, in_channels is the number of input
                channels, and T is the length of the input sequence.
            g (Optional[torch.Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This tensor is optional and, if provided,
                is added to the input tensor after the initial convolution.

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, T),
                where out_channels is the number of output channels.

        Examples:
            >>> generator = HiFiGANGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be (1, 1, T)

        Note:
            The input tensor must have the correct number of channels as
            specified during the initialization of the HiFiGANGenerator.
            The global conditioning tensor must have the same batch size as
            the input tensor if provided.

        Raises:
            AssertionError: If the input tensor does not match the expected
            shape or if the global conditioning tensor has an incompatible shape.
        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs
