# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""MelGAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from espnet2.gan_tts.melgan.residual_stack import ResidualStack


class MelGANGenerator(torch.nn.Module):
    """
        MelGAN generator module.

    This module implements the MelGAN generator architecture, which is designed for
    generating audio waveforms from Mel spectrograms. It utilizes a series of
    convolutional layers, upsampling layers, and residual connections to produce
    high-quality audio outputs.

    Attributes:
        melgan (torch.nn.Sequential): The sequential model comprising various
            layers for processing the input tensor.

    Args:
        in_channels (int): Number of input channels (default: 80).
        out_channels (int): Number of output channels (default: 1).
        kernel_size (int): Kernel size of initial and final conv layer (default: 7).
        channels (int): Initial number of channels for conv layer (default: 512).
        bias (bool): Whether to add bias parameter in convolution layers (default: True).
        upsample_scales (List[int]): List of upsampling scales (default: [8, 8, 2, 2]).
        stack_kernel_size (int): Kernel size of dilated conv layers in residual stack
            (default: 3).
        stacks (int): Number of stacks in a single residual stack (default: 3).
        nonlinear_activation (str): Activation function module name (default: "LeakyReLU").
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
            function (default: {"negative_slope": 0.2}).
        pad (str): Padding function module name before dilated convolution layer
            (default: "ReflectionPad1d").
        pad_params (Dict[str, Any]): Hyperparameters for padding function (default: {}).
        use_final_nonlinear_activation (bool): Whether to use final activation function
            (default: True).
        use_weight_norm (bool): Whether to use weight normalization (default: True).

    Raises:
        AssertionError: If hyperparameters are invalid, such as the number of channels
            or kernel size.

    Examples:
        >>> generator = MelGANGenerator(in_channels=80, out_channels=1)
        >>> input_tensor = torch.randn(1, 80, 100)  # Batch size of 1, 80 channels, length 100
        >>> output_tensor = generator(input_tensor)
        >>> print(output_tensor.shape)  # Should output: (1, 1, 1600) based on upsampling scales
    """

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        kernel_size: int = 7,
        channels: int = 512,
        bias: bool = True,
        upsample_scales: List[int] = [8, 8, 2, 2],
        stack_kernel_size: int = 3,
        stacks: int = 3,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
        use_final_nonlinear_activation: bool = True,
        use_weight_norm: bool = True,
    ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (List[int]): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual
                stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for
                the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
        ]

        self.upsample_factor = int(np.prod(upsample_scales) * out_channels)
        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            layers += [
                torch.nn.ConvTranspose1d(
                    channels // (2**i),
                    channels // (2 ** (i + 1)),
                    upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    bias=bias,
                )
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size**j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                    )
                ]

        # add final layer
        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias
            ),
        ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method computes the forward pass of the MelGAN generator. It takes
        an input tensor and processes it through the generator network to produce
        an output tensor.

        Args:
            c (Tensor): Input tensor of shape (B, channels, T), where:
                - B is the batch size.
                - channels is the number of input channels.
                - T is the sequence length.

        Returns:
            Tensor: Output tensor of shape (B, 1, T ** prod(upsample_scales)),
            where:
                - The output tensor has a single channel and its length is the
                result of upsampling the input sequence.

        Examples:
            >>> generator = MelGANGenerator()
            >>> input_tensor = torch.randn(8, 80, 100)  # Example input
            >>> output_tensor = generator(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([8, 1, 6400])  # Example output shape
        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.

        This method iterates through all layers of the MelGAN generator and
        removes weight normalization from each layer that has it applied. Weight
        normalization is a technique used to stabilize the training of deep
        networks, but there may be cases where it is desirable to remove it.

        It utilizes the `torch.nn.utils.remove_weight_norm` function, which
        raises a ValueError if the module does not have weight normalization
        applied. This method handles that exception and logs the removal of
        weight normalization.

        Examples:
            >>> generator = MelGANGenerator()
            >>> generator.apply_weight_norm()  # Apply weight normalization first
            >>> generator.remove_weight_norm()  # Now remove weight normalization

        Note:
            This function modifies the internal state of the model. Ensure that
            the model is in the appropriate state before calling this method.
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

        This method iterates through all the layers of the model and applies
        weight normalization to each convolutional layer. Weight normalization
        can help in stabilizing the training of deep networks by improving the
        conditioning of the optimization problem.

        Note:
            This method is called during the initialization of the model if
            `use_weight_norm` is set to `True`.

        Examples:
            >>> model = MelGANGenerator(use_weight_norm=True)
            >>> model.apply_weight_norm()
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

        This method reinitializes the weights of the convolutional layers in the
        MelGAN generator according to the official implementation. It uses a
        normal distribution with a mean of 0 and a standard deviation of 0.02
        for the weights of the Conv1d and ConvTranspose1d layers. This can be
        useful for ensuring that the model starts training with reasonable
        weight values.

        This initialization follows the official implementation manner as
        described in the following link:
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        Examples:
            >>> generator = MelGANGenerator()
            >>> generator.reset_parameters()  # Reinitialize weights of layers
        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c: torch.Tensor) -> torch.Tensor:
        """
        Perform inference.

        This method processes the input tensor through the MelGAN generator to
        produce an output tensor, which can be used for audio synthesis.

        Args:
            c (Tensor): Input tensor of shape (T, in_channels) where T is the
                length of the input sequence and in_channels is the number of
                input channels (typically the number of Mel frequency bands).

        Returns:
            Tensor: Output tensor of shape (T ** prod(upsample_scales),
            out_channels) where out_channels is the number of output channels
            (typically 1 for mono audio).

        Examples:
            >>> generator = MelGANGenerator()
            >>> mel_input = torch.randn(100, 80)  # Example input tensor
            >>> output = generator.inference(mel_input)
            >>> print(output.shape)  # Should print (800, 1) if upsample_scales = [8, 8, 2, 2]
        """
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class MelGANDiscriminator(torch.nn.Module):
    """
        MelGAN discriminator module.

    This class implements the discriminator used in the MelGAN architecture. It is
    designed to process audio signals, learning to distinguish between real and
    generated samples.

    Attributes:
        layers (torch.nn.ModuleList): A list of layers that constitute the discriminator.

    Args:
        in_channels (int): Number of input channels. Default is 1.
        out_channels (int): Number of output channels. Default is 1.
        kernel_sizes (List[int]): List of two kernel sizes. The product will be used
            for the first conv layer, and the first and the second kernel sizes
            will be used for the last two layers. For example if kernel_sizes =
            [5, 3], the first layer kernel size will be 5 * 3 = 15, the last two
            layers' kernel size will be 5 and 3, respectively.
        channels (int): Initial number of channels for conv layer. Default is 16.
        max_downsample_channels (int): Maximum number of channels for downsampling
            layers. Default is 1024.
        bias (bool): Whether to add bias parameter in convolution layers. Default is True.
        downsample_scales (List[int]): List of downsampling scales. Default is [4, 4, 4, 4].
        nonlinear_activation (str): Activation function module name. Default is "LeakyReLU".
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
            function. Default is {"negative_slope": 0.2}.
        pad (str): Padding function module name before dilated convolution layer.
            Default is "ReflectionPad1d".
        pad_params (Dict[str, Any]): Hyperparameters for padding function. Default is {}.

    Examples:
        >>> discriminator = MelGANDiscriminator()
        >>> input_tensor = torch.randn(2, 1, 16000)  # (B, 1, T)
        >>> output = discriminator(input_tensor)
        >>> len(output)  # Should return the number of layers in the discriminator

    Returns:
        List[Tensor]: List of output tensors of each layer.

    Raises:
        AssertionError: If the kernel sizes are not valid (not odd).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: List[int] = [5, 3],
        channels: int = 16,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        downsample_scales: List[int] = [4, 4, 4, 4],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
    ):
        """Initilize MelGANDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (List[int]): List of two kernel sizes. The prod will be used
                for the first conv layer, and the first and the second kernel sizes
                will be used for the last two layers. For example if kernel_sizes =
                [5, 3], the first layer kernel size will be 5 * 3 = 15, the last two
                layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling
                layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    in_channels, channels, np.prod(kernel_sizes), bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate forward propagation.

        This method performs the forward pass of the MelGAN discriminator. It takes
        an input tensor and processes it through the defined layers of the model,
        returning the output tensor.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        Examples:
            >>> model = MelGANGenerator()
            >>> input_tensor = torch.randn(16, 80, 100)  # Batch size of 16
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([16, 1, 1600])  # Example output shape based on upsampling
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """
        MelGAN multi-scale discriminator module.

    This class implements a multi-scale discriminator for the MelGAN architecture,
    allowing the model to evaluate audio signals at different resolutions. The
    discriminator consists of multiple MelGANDiscriminator instances that process
    the input through a specified downsampling pooling operation.

    Attributes:
        discriminators (torch.nn.ModuleList): A list of MelGANDiscriminator
            instances, one for each scale.
        pooling (torch.nn.Module): The pooling layer used for downsampling the
            input signal.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scales (int): Number of multi-scales.
        downsample_pooling (str): Pooling module name for downsampling of the
            inputs.
        downsample_pooling_params (Dict[str, Any]): Parameters for the above
            pooling module.
        kernel_sizes (List[int]): List of two kernel sizes. The sum will be used
            for the first conv layer, and the first and the second kernel sizes
            will be used for the last two layers.
        channels (int): Initial number of channels for conv layer.
        max_downsample_channels (int): Maximum number of channels for
            downsampling layers.
        bias (bool): Whether to add bias parameter in convolution layers.
        downsample_scales (List[int]): List of downsampling scales.
        nonlinear_activation (str): Activation function module name.
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
            function.
        pad (str): Padding function module name before dilated convolution layer.
        pad_params (Dict[str, Any]): Hyperparameters for padding function.
        use_weight_norm (bool): Whether to use weight norm.

    Examples:
        >>> discriminator = MelGANMultiScaleDiscriminator(in_channels=1,
        ...     out_channels=1, scales=3)
        >>> input_tensor = torch.randn(8, 1, 1024)  # Batch of 8, 1 channel, 1024 length
        >>> outputs = discriminator(input_tensor)
        >>> len(outputs)  # Number of scales
        3

    Returns:
        List[List[Tensor]]: List of lists containing the outputs from each
            discriminator for each scale.

    Raises:
        AssertionError: If any of the hyperparameters are invalid.

    Note:
        This class follows the official implementation manner for initializing
        parameters as described in the original MelGAN repository.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scales: int = 3,
        downsample_pooling: str = "AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes: List[int] = [5, 3],
        channels: int = 16,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        downsample_scales: List[int] = [4, 4, 4, 4],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
        use_weight_norm: bool = True,
    ):
        """Initilize MelGANMultiScaleDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            downsample_pooling_params (Dict[str, Any]): Parameters for the above
                pooling module.
            kernel_sizes (List[int]): List of two kernel sizes. The sum will be used
                for the first conv layer, and the first and the second kernel sizes
                will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling
                layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.
            use_weight_norm (bool): Whether to use weight norm.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Calculate forward propagation.

        This method takes an input tensor and passes it through the multi-scale
        discriminators, returning the outputs of each layer for all discriminators.

        Args:
            x (Tensor): Input noise signal (B, 1, T), where B is the batch size,
                1 is the number of input channels, and T is the length of the
                input signal.

        Returns:
            List[List[Tensor]]: A list of lists, where each inner list contains
                the output tensors from each layer of a discriminator. The outer
                list corresponds to the outputs from each discriminator in the
                multi-scale setup.

        Examples:
            >>> discriminator = MelGANMultiScaleDiscriminator()
            >>> input_tensor = torch.randn(4, 1, 1024)  # Batch of 4, 1 channel, T=1024
            >>> outputs = discriminator(input_tensor)
            >>> len(outputs)  # Should be equal to the number of scales
            3
            >>> len(outputs[0])  # Should match the number of layers in the discriminator
            5  # For example, if each discriminator has 5 layers
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.

        This method iterates through all layers of the MelGAN multi-scale
        discriminator and removes the weight normalization applied to the
        convolutional layers. If a layer does not have weight normalization,
        it will be skipped without raising an error.

        Example:
            >>> discriminator = MelGANMultiScaleDiscriminator()
            >>> discriminator.apply_weight_norm()  # Apply weight normalization
            >>> discriminator.remove_weight_norm()  # Remove weight normalization

        Note:
            This method is useful when switching between training and
            evaluation modes where weight normalization might be needed
            only during training.
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

        This method iterates through all the layers of the MelGANMultiScaleDiscriminator
        and applies weight normalization to each convolutional layer. Weight normalization
        can help stabilize the training of neural networks by reparameterizing the weight
        vectors.

        It specifically targets layers of type `torch.nn.Conv1d` and
        `torch.nn.ConvTranspose1d`, applying the weight normalization technique from
        `torch.nn.utils`.

        Examples:
            >>> discriminator = MelGANMultiScaleDiscriminator()
            >>> discriminator.apply_weight_norm()  # Applies weight normalization

        Note:
            This function is usually called during the initialization of the model
            if `use_weight_norm` is set to True.
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
        Reset parameters of the model.

        This method reinitializes the weights of the convolutional layers in the
        MelGAN model according to the official implementation guidelines. It sets
        the weights of each convolutional layer to a normal distribution with a
        mean of 0 and a standard deviation of 0.02.

        This follows the initialization method specified in the official MelGAN
        implementation:
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        It is important to call this method to ensure that the model parameters
        are in a known state, especially after loading a pre-trained model or
        modifying the architecture.

        Examples:
            >>> model = MelGANMultiScaleDiscriminator()
            >>> model.reset_parameters()  # Resets all parameters in the model
        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
