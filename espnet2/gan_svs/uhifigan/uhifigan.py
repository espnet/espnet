# -*- coding: utf-8 -*-

"""Unet-baed HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan
and https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

try:
    from parallel_wavegan.layers import CausalConv1d, CausalConvTranspose1d
    from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
    from parallel_wavegan.utils import read_hdf5
except ImportError:
    CausalConv1d, CausalConvTranspose1d = None, None
    ResidualBlock = None
    read_hdf5 = None


class UHiFiGANGenerator(torch.nn.Module):
    """
    UHiFiGAN generator module.

    This class implements a Unet-based generator for the HiFi-GAN model.
    It is designed to perform high-fidelity audio synthesis based on input
    mel-spectrograms and additional conditioning information. The architecture
    is inspired by existing implementations in the HiFi-GAN and ParallelWaveGAN
    repositories.

    This code is based on:
        - https://github.com/jik876/hifi-gan
        - https://github.com/kan-bayashi/ParallelWaveGAN

    Attributes:
        input_conv (nn.Sequential): Initial convolutional layer.
        downsamples (nn.ModuleList): List of downsampling layers.
        downsamples_mrf (nn.ModuleList): List of multi-resolution
            feature extraction layers for downsampling.
        hidden_conv (nn.Conv1d): Hidden convolutional layer.
        upsamples (nn.ModuleList): List of upsampling layers.
        upsamples_mrf (nn.ModuleList): List of multi-resolution
            feature extraction layers for upsampling.
        output_conv (nn.ModuleList): Final convolutional layers.
        global_conv (nn.Conv1d): Global conditioning layer if global channels
            are specified.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of hidden representation channels.
        global_channels (int): Number of global conditioning channels.
        kernel_size (int): Kernel size of initial and final conv layer.
        downsample_scales (tuple): List of downsampling scales.
        downsample_kernel_sizes (tuple): List of kernel sizes for downsampling
            layers.
        upsample_scales (tuple): List of upsampling scales.
        upsample_kernel_sizes (tuple): List of kernel sizes for upsampling layers.
        resblock_kernel_sizes (tuple): List of kernel sizes for residual blocks.
        resblock_dilations (list): List of dilation lists for residual blocks.
        projection_filters (list): List of projection filters.
        projection_kernels (list): List of projection kernel sizes.
        dropout (float): Dropout rate.
        use_additional_convs (bool): Whether to use additional conv layers
            in residual blocks.
        bias (bool): Whether to add bias parameter in convolution layers.
        nonlinear_activation (str): Activation function module name.
        nonlinear_activation_params (dict): Hyperparameters for activation function.
        use_causal_conv (bool): Whether to use causal structure.
        use_weight_norm (bool): Whether to use weight normalization.
        use_avocodo (bool): Whether to use Avocodo structure for output.

    Raises:
        ImportError: If the required modules from `parallel_wavegan` are not
        installed.

    Examples:
        >>> generator = UHiFiGANGenerator()
        >>> c = torch.randn(1, 80, 100)  # Example input tensor
        >>> f0 = torch.randn(1, 1, 100)   # Example f0 tensor
        >>> excitation = torch.randn(1, 512, 100)  # Example excitation tensor
        >>> output = generator(c, f0, excitation)
        >>> print(output.shape)  # Output tensor shape

    Note:
        The model is based on the HiFi-GAN architecture, which is optimized
        for high-quality audio generation. The initialization of weights follows
        the official implementation.

    Todo:
        - Add support for additional input modalities.
    """

    @typechecked
    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        global_channels: int = -1,
        kernel_size=7,
        downsample_scales=(2, 2, 8, 8),
        downsample_kernel_sizes=(4, 4, 16, 16),
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        projection_filters: List[int] = [0, 1, 1, 1],
        projection_kernels: List[int] = [0, 5, 7, 11],
        dropout=0.3,
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_causal_conv=False,
        use_weight_norm=True,
        use_avocodo=False,
    ):
        """Initialize Unet-based HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            global_channels (int): Number of global conditioning channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers
                in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_causal_conv (bool): Whether to use causal structure.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.use_causal_conv = use_causal_conv

        self.input_conv = None

        self.downsamples = torch.nn.ModuleList()
        self.downsamples_mrf = torch.nn.ModuleList()

        self.hidden_conv = None

        self.upsamples = torch.nn.ModuleList()
        self.upsamples_mrf = torch.nn.ModuleList()

        self.output_conv = None
        self.use_avocodo = use_avocodo

        if (
            CausalConv1d is None
            or CausalConvTranspose1d is None
            or ResidualBlock is None
        ):
            raise ImportError(
                "`parallel_wavegan` is not installed. "
                "Please install via `pip install -U parallel_wavegan`."
            )

        if not use_causal_conv:
            self.input_conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    out_channels,
                    channels,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout),
            )
        else:
            self.input_conv = torch.nn.Sequential(
                CausalConv1d(
                    out_channels,
                    channels,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout),
            )

        for i in range(len(downsample_scales)):
            for j in range(len(resblock_kernel_sizes)):
                self.downsamples_mrf += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels,
                        # channels=channels * 2**i,
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            if not use_causal_conv:
                self.downsamples += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            channels,
                            channels * 2,
                            # channels * (2 ** (i + 1)),
                            kernel_size=downsample_kernel_sizes[i],
                            stride=downsample_scales[i],
                            bias=bias,
                            padding=downsample_scales[i] // 2
                            + downsample_scales[i] % 2,
                        ),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Dropout(dropout),
                    )
                ]
            else:
                self.downsamples += [
                    torch.nn.Sequential(
                        CausalConv1d(
                            channels,
                            channels * 2,
                            # channels * (2 ** (i + 1)),
                            kernel_size=downsample_kernel_sizes[i],
                            stride=downsample_scales[i],
                            bias=bias,
                            padding=downsample_scales[i] // 2
                            + downsample_scales[i] % 2,
                        ),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Dropout(dropout),
                    )
                ]

            channels = channels * 2

        if not use_causal_conv:
            self.hidden_conv = torch.nn.Conv1d(
                in_channels,
                channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.hidden_conv = CausalConv1d(
                in_channels,
                channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )

        max_channels = channels
        self.output_conv = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            # assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            if not use_causal_conv:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.ConvTranspose1d(
                            channels * 2,
                            channels // 2,
                            # channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                ]
            else:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        CausalConvTranspose1d(
                            channels * 2,
                            channels // 2,
                            # channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            bias=bias,
                        ),
                    )
                ]

            # hidden_channel for MRF module
            for j in range(len(resblock_kernel_sizes)):
                self.upsamples_mrf += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // 2,
                        # channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            channels = channels // 2

            if use_avocodo:
                if projection_filters[i] != 0:
                    self.output_conv.append(
                        torch.nn.Conv1d(
                            max_channels // (2 ** (i + 1)),
                            # channels // (2 ** (i + 1)),
                            projection_filters[i],
                            projection_kernels[i],
                            1,
                            padding=projection_kernels[i] // 2,
                        )
                    )
                else:
                    self.output_conv.append(torch.nn.Identity())
        if not use_avocodo:
            if not use_causal_conv:
                self.output_conv = torch.nn.Sequential(
                    # NOTE(kan-bayashi): follow official implementation but why
                    #   using different slope parameter here? (0.1 vs. 0.01)
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        channels,
                        out_channels,
                        kernel_size,
                        bias=bias,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.Tanh(),
                )
            else:
                self.output_conv = torch.nn.Sequential(
                    # NOTE(kan-bayashi): follow official implementation but why
                    #   using different slope parameter here? (0.1 vs. 0.01)
                    torch.nn.LeakyReLU(),
                    CausalConv1d(
                        channels,
                        out_channels,
                        kernel_size,
                        bias=bias,
                    ),
                    torch.nn.Tanh(),
                )

        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, in_channels, 1)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(
        self, c=None, f0=None, excitation=None, g: Optional[torch.Tensor] = None
    ):
        """
        Calculate forward propagation.

        This method performs the forward pass of the UHiFiGAN generator. It
        takes input tensors, processes them through the network layers, and
        returns the output tensor.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            f0 (Tensor): Input tensor (B, 1, T).
            excitation (Tensor): Input tensor (B, frame_len, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, T).
                This is optional and can be provided to enhance the input tensor
                `c`.

        Returns:
            Tensor: Output tensor (B, out_channels, T). This tensor represents the
            generated audio signal.

        Examples:
            >>> generator = UHiFiGANGenerator()
            >>> c = torch.randn(2, 80, 100)  # Example input tensor
            >>> f0 = torch.randn(2, 1, 100)   # Example f0 tensor
            >>> excitation = torch.randn(2, 256, 100)  # Example excitation tensor
            >>> output = generator.forward(c, f0, excitation)
            >>> print(output.shape)
            torch.Size([2, 1, 100])  # Output shape matches the expected output tensor
        """
        # logging.warn(f'c:{c.shape}')
        # logging.warn(f'f0:{f0.shape}')
        # logging.warn(f'excitation:{excitation.shape}')

        # logging.info(f'c:{c.shape}')
        # if f0 is not None:
        #     c = torch.cat( (c,f0), 1)
        # if excitation is not None:
        #     c = torch.cat( (c,excitation), 1)
        # if f0 is not None and excitation is not None:
        #     c = torch.cat( (c, f0, excitation) ,1)
        # elif f0 is not None:
        #     c = torch.cat( (c,f0), 1)
        # elif excitation is not None:
        #     c = torch.cat( (c,excitation), 1)

        residual_results = []
        if self.use_avocodo:
            outs = []
        hidden = self.input_conv(excitation)

        # TODO(yifeng): add global conv to hidden?
        if g is not None:
            c = c + self.global_conv(g)

        for i in range(len(self.downsamples)):
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                tc = self.downsamples_mrf[i * self.num_blocks + j](hidden)
                cs += tc
            hidden = cs / self.num_blocks
            hidden = self.downsamples[i](hidden)
            # print(f"hidden.shape{i}", hidden.shape)
            residual_results.append(hidden)

        # logging.warn(f'hidden:{hidden.shape}')
        residual_results.reverse()
        # logging.warn(f"residual_results:{ [r.shape for r in residual_results] }")

        hidden_mel = self.hidden_conv(c)

        for i in range(len(self.upsamples)):
            # logging.warn(f'bef {i}-th upsampe:{hidden_mel.shape}')
            # logging.warn(f'bef {i}-th upsampe:{residual_results[i].shape}')
            # print("hidden_mel.shape1", hidden_mel.shape)
            hidden_mel = torch.cat((hidden_mel, residual_results[i]), dim=1)
            # logging.warn(f'aft {i}-th upsample :{hidden_mel.shape}')
            # print("hidden_mel.shape2", hidden_mel.shape)
            hidden_mel = self.upsamples[i](hidden_mel)
            # print("hidden_mel.shape3", hidden_mel.shape)
            # logging.warn(f'bef {i}-th MRF:{hidden_mel.shape}')
            # logging.warn(f'self.upsamples_mrf:{self.upsamples_mrf}')
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                tc = self.upsamples_mrf[i * self.num_blocks + j](hidden_mel)
                # logging.info(f'{j}-th tc.shape:{tc.shape}')
                cs += tc
            hidden_mel = cs / self.num_blocks
            # logging.warn(f'aft {i}-th MRF:{hidden_mel.shape}')
            if self.use_avocodo:
                if i >= (self.num_upsamples - 3):
                    _c = F.leaky_relu(hidden_mel)
                    _c = self.output_conv[i](_c)
                    _c = torch.tanh(_c)
                    outs.append(_c)
                else:
                    hidden_mel = self.output_conv[i](hidden_mel)

        # logging.warn(f'bef output conv mel : {hidden_mel.shape}')
        if self.use_avocodo:
            return outs
        else:
            return self.output_conv(hidden_mel)

    def reset_parameters(self):
        """
        Reset parameters.

        This initialization follows the official implementation manner.
        The weights of convolutional layers are initialized with a normal
        distribution having a mean of 0.0 and a standard deviation of 0.01.
        This method is called during the initialization of the generator
        to ensure that all parameters are set to a reasonable starting
        point for training.

        This method applies to all layers of type `Conv1d` and
        `ConvTranspose1d`.

        Example:
            >>> generator = UHiFiGANGenerator()
            >>> generator.reset_parameters()  # Resets the weights of the model

        Note:
            This function is part of the initialization routine and should
            be called after defining the model architecture but before
            training the model.
        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """
        Remove weight normalization from all convolutional layers.

        This method traverses all layers of the UHiFiGANGenerator and removes
        weight normalization if it has been applied. It is useful when weight
        normalization is no longer needed, for instance, when switching to a
        different training phase or during inference.

        Note:
            This function will log a message each time weight normalization
            is removed from a layer. If a layer does not have weight normalization,
            a ValueError is caught and ignored.

        Examples:
            >>> generator = UHiFiGANGenerator(use_weight_norm=True)
            >>> generator.remove_weight_norm()  # Removes weight norm from layers.
        """

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.

        This method iterates through all layers of the model and applies weight
        normalization to each layer that is an instance of `torch.nn.Conv1d` or
        `torch.nn.ConvTranspose1d`. Weight normalization can help stabilize the
        training of neural networks by normalizing the weights during forward
        propagation.

        Note:
            Weight normalization is beneficial for improving convergence speed
            and can lead to better overall performance in some cases.

        Example:
            >>> generator = UHiFiGANGenerator(use_weight_norm=True)
            >>> generator.apply_weight_norm()
            # This will apply weight normalization to all convolutional layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def register_stats(self, stats):
        """
        Register stats for de-normalization as buffer.

        This method loads statistical data from a specified file and registers
        them as buffers for normalization purposes. The file can be either in
        HDF5 or NumPy format, containing mean and scale values for
        de-normalization.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").
                The file should contain two arrays: mean and scale.

        Raises:
            AssertionError: If the file does not have a valid extension
                (".h5" or ".npy").

        Examples:
            >>> generator = UHiFiGANGenerator()
            >>> generator.register_stats("path/to/stats.npy")
            Successfully registered stats as buffer.
            >>> generator.register_stats("path/to/stats.h5")
            Successfully registered stats as buffer.

        Note:
            The mean and scale values are expected to be one-dimensional
            arrays. The method reshapes them into a suitable format for
            registration as buffers.
        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, excitation=None, f0=None, c=None, normalize_before=False):
        """
        Perform inference using the UHiFiGAN generator.

        This method takes the input tensors for excitation, fundamental frequency
        (f0), and conditioning (c), and performs a forward pass through the
        UHiFiGAN generator. It outputs a generated waveform tensor.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels). This tensor
                represents the conditioning information used during inference.
            excitation (Union[Tensor, ndarray], optional): Input tensor for
                excitation (T, frame_len). This tensor is used to control the
                excitation signal in the generation process.
            f0 (Union[Tensor, ndarray], optional): Input tensor for fundamental
                frequency (T, 1). This tensor provides information about the
                pitch of the audio to be generated.
            normalize_before (bool, optional): Whether to perform normalization
                before the inference. Defaults to False.

        Returns:
            Tensor: Output tensor of shape (T ** prod(upsample_scales),
            out_channels), representing the generated audio waveform.

        Examples:
            >>> generator = UHiFiGANGenerator()
            >>> excitation = torch.randn(1, 80, 256)  # Example excitation
            >>> f0 = torch.randn(1, 1, 256)            # Example f0
            >>> c = torch.randn(256, 80)                # Example conditioning
            >>> output = generator.inference(excitation, f0, c)

        Note:
            The input tensors can be either PyTorch tensors or NumPy arrays. If
            they are not tensors, they will be converted to tensors before processing.
        """
        # print(len(c))
        # logging.info(f'len(c):{len(c)}')
        # excitation, f0, c = c
        if c is not None and not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if excitation is not None and not isinstance(excitation, torch.Tensor):
            excitation = torch.tensor(excitation, dtype=torch.float).to(
                next(self.parameters()).device
            )
        if f0 is not None and not isinstance(f0, torch.Tensor):
            f0 = torch.tensor(f0, dtype=torch.float).to(next(self.parameters()).device)

        # logging.info(f'excitation.shape:{excitation.shape}')
        # logging.info(f'f0.shape:{f0.shape}')
        # logging.info(f'c.shape:{c.shape}')
        # c = self.forward(None, None, c.transpose(1, 0).unsqueeze(0))
        c = self.forward(
            c.transpose(1, 0).unsqueeze(0),
            f0.unsqueeze(1).transpose(1, 0).unsqueeze(0),
            excitation.reshape(1, 1, -1),
        )
        return c.squeeze(0).transpose(1, 0)
