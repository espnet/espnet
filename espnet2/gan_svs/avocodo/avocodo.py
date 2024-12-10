# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Avocodo Modules.

This code is modified from https://github.com/ncsoft/avocodo.

"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import spectral_norm, weight_norm

from espnet2.gan_svs.visinger2.visinger2_vocoder import MultiFrequencyDiscriminator
from espnet2.gan_tts.hifigan.residual_block import ResidualBlock
from espnet2.gan_tts.melgan.pqmf import PQMF


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding needed for a convolution operation.

    This function computes the amount of padding required to maintain the
    spatial dimensions of the input tensor when applying a convolution
    with a specified kernel size and dilation.

    Args:
        kernel_size (int): The size of the convolution kernel.
        dilation (int, optional): The dilation rate for the convolution.
            Default is 1, which means no dilation.

    Returns:
        int: The calculated padding value.

    Examples:
        >>> get_padding(3)
        1
        >>> get_padding(5, dilation=2)
        4
        >>> get_padding(7, dilation=3)
        9
    """
    return int((kernel_size * dilation - dilation) / 2)


class AvocodoGenerator(torch.nn.Module):
    """
    Avocodo generator module for generating audio signals.

    This module utilizes various convolutional layers and residual blocks
    to generate audio signals from input features, allowing for multi-scale
    and multi-resolution processing.

    Attributes:
        num_upsamples (int): Number of upsampling layers.
        num_blocks (int): Number of residual blocks.
        input_conv (Conv1d): Initial convolutional layer.
        upsamples (ModuleList): List of upsampling layers.
        blocks (ModuleList): List of residual blocks.
        output_conv (ModuleList): List of output convolutional layers.
        global_conv (Optional[Conv1d]): Global conditioning convolutional layer.

    Args:
        in_channels (int): Number of input channels. Defaults to 80.
        out_channels (int): Number of output channels. Defaults to 1.
        channels (int): Number of hidden representation channels. Defaults to 512.
        global_channels (int): Number of global conditioning channels. Defaults to -1.
        kernel_size (int): Kernel size of initial and final conv layer. Defaults to 7.
        upsample_scales (List[int]): List of upsampling scales. Defaults to [8, 8, 2, 2].
        upsample_kernel_sizes (List[int]): List of kernel sizes for upsample layers.
            Defaults to [16, 16, 4, 4].
        resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks.
            Defaults to [3, 7, 11].
        resblock_dilations (List[List[int]]): List of list of dilations for residual
            blocks. Defaults to [[1, 3, 5], [1, 3, 5], [1, 3, 5]].
        projection_filters (List[int]): List of projection filters. Defaults to
            [0, 1, 1, 1].
        projection_kernels (List[int]): List of projection kernels. Defaults to
            [0, 5, 7, 11].
        use_additional_convs (bool): Whether to use additional conv layers in
            residual blocks. Defaults to True.
        bias (bool): Whether to add bias parameter in convolution layers. Defaults to True.
        nonlinear_activation (str): Activation function module name. Defaults to "LeakyReLU".
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
            function. Defaults to {"negative_slope": 0.2}.
        use_weight_norm (bool): Whether to use weight norm. Defaults to True.

    Raises:
        AssertionError: If kernel size is not odd or the lengths of
            upsample parameters do not match.

    Examples:
        >>> generator = AvocodoGenerator(in_channels=80, out_channels=1)
        >>> input_tensor = torch.randn(1, 80, 100)  # Batch size of 1, 80 channels, length 100
        >>> output = generator(input_tensor)
        >>> print([o.shape for o in output])  # Output shapes for each upsampled tensor
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
        projection_filters: List[int] = [0, 1, 1, 1],
        projection_kernels: List[int] = [0, 5, 7, 11],
        use_additional_convs: bool = True,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        use_weight_norm: bool = True,
    ):
        """Initialize AvocodoGenerator module.

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
        self.output_conv = torch.nn.ModuleList()
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

            if projection_filters[i] != 0:
                self.output_conv.append(
                    torch.nn.Conv1d(
                        channels // (2 ** (i + 1)),
                        projection_filters[i],
                        projection_kernels[i],
                        1,
                        padding=projection_kernels[i] // 2,
                    )
                )
            else:
                self.output_conv.append(torch.nn.Identity())

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

        This method computes the forward pass of the AvocodoGenerator. It takes an
        input tensor and, optionally, a global conditioning tensor. The output is
        a list of output tensors generated from the input.

        Args:
            c (Tensor): Input tensor of shape (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). If provided, it will be added to the
                input tensor after the initial convolution.

        Returns:
            List[Tensor]: List of output tensors of shape (B, out_channels, T).

        Examples:
            >>> generator = AvocodoGenerator()
            >>> input_tensor = torch.randn(2, 80, 100)  # Batch size 2, 80 channels
            >>> global_conditioning = torch.randn(2, 10, 1)  # Batch size 2, 10 channels
            >>> outputs = generator(input_tensor, global_conditioning)
            >>> for output in outputs:
            ...     print(output.shape)
            torch.Size([2, 1, 100])  # Output shape for each output tensor

        Note:
            The number of outputs in the returned list corresponds to the number of
            upsampling layers defined during initialization.
        """
        outs = []
        c = self.input_conv(c)
        if g is not None:
            c = c + self.global_conv(g)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
            if i >= (self.num_upsamples - 3):
                _c = F.leaky_relu(c)
                _c = self.output_conv[i](_c)
                _c = torch.tanh(_c)
                outs.append(_c)
            else:
                c = self.output_conv[i](c)

        return outs

    def reset_parameters(self):
        """
        Reset parameters.

        This method initializes the weights of the convolutional layers in the
        generator according to the official implementation manner described in
        the HiFi-GAN repository:
        https://github.com/jik876/hifi-gan/blob/master/models.py. The weights
        are drawn from a normal distribution with a mean of 0 and a standard
        deviation of 0.01.

        This method is called during the initialization of the generator
        to ensure that all layers start with appropriate weights.

        Note:
            The logging module is used to output debug information when
            resetting parameters for each layer.

        Examples:
            >>> generator = AvocodoGenerator()
            >>> generator.reset_parameters()  # Reset parameters to initial values
        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.

        This method iterates through all layers of the `AvocodoGenerator`
        and removes weight normalization if it has been applied. It will
        log a debug message each time weight normalization is removed from
        a layer. If a layer does not have weight normalization, it will
        catch the `ValueError` and continue without interruption.

        Examples:
            >>> generator = AvocodoGenerator(use_weight_norm=True)
            >>> generator.remove_weight_norm()  # Removes weight normalization
            >>> # Subsequent calls to generator will not use weight normalization.

        Note:
            This method is particularly useful when fine-tuning or modifying
            the model's architecture after training.
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

        This method applies weight normalization to all convolutional layers
        (Conv1d and ConvTranspose1d) within the AvocodoGenerator module. Weight
        normalization helps in stabilizing the training of deep neural networks
        by reparameterizing the weight vectors, which can improve convergence
        speed and model performance.

        This method is automatically called during the initialization of the
        AvocodoGenerator class if the `use_weight_norm` parameter is set to
        True.

        Examples:
            # Create an instance of AvocodoGenerator with weight normalization
            generator = AvocodoGenerator(use_weight_norm=True)

            # Create an instance without weight normalization
            generator_no_norm = AvocodoGenerator(use_weight_norm=False)

        Note:
            This method should be called only after the model has been fully
            constructed and the layers have been defined.
        """

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


# CoMBD
class CoMBDBlock(torch.nn.Module):
    """
    CoMBD (Collaborative Multi-band Discriminator) block module.

    This module implements a collaborative multi-band discriminator block that
    processes input signals through a series of convolutional layers. The design
    allows for different configurations of kernel sizes, strides, dilations, and
    groups for each convolutional layer, enabling flexibility in the architecture.

    Attributes:
        convs (torch.nn.ModuleList): A list of convolutional layers defined by the
            input parameters.
        projection_conv (torch.nn.Module): A final convolutional layer for output
            projection.

    Args:
        h_u (List[int]): List of hidden units for each layer.
        d_k (List[int]): List of kernel sizes for each convolutional layer.
        d_s (List[int]): List of strides for each convolutional layer.
        d_d (List[int]): List of dilations for each convolutional layer.
        d_g (List[int]): List of groups for each convolutional layer.
        d_p (List[int]): List of paddings for each convolutional layer.
        op_f (int): Number of output filters for the final projection layer.
        op_k (int): Kernel size for the final projection layer.
        op_g (int): Number of groups for the final projection layer.
        use_spectral_norm (bool): Whether to apply spectral normalization to the
            convolutional layers.

    Returns:
        None

    Examples:
        >>> block = CoMBDBlock(
        ...     h_u=[16, 64, 256],
        ...     d_k=[3, 5, 7],
        ...     d_s=[1, 2, 1],
        ...     d_d=[1, 1, 1],
        ...     d_g=[1, 1, 1],
        ...     d_p=[1, 2, 3],
        ...     op_f=1,
        ...     op_k=3,
        ...     op_g=1,
        ...     use_spectral_norm=True
        ... )
        >>> input_tensor = torch.randn(1, 16, 1024)  # Example input
        >>> output, feature_maps = block(input_tensor)

    Returns:
        Tuple[Tensor, List[Tensor]]: Tuple containing the output tensor of shape
        (B, C_out, T_out) and a list of feature maps of shape (B, C, T) at each
        Conv1d layer.
    """

    def __init__(
        self,
        h_u: List[int],
        d_k: List[int],
        d_s: List[int],
        d_d: List[int],
        d_g: List[int],
        d_p: List[int],
        op_f: int,
        op_k: int,
        op_g: int,
        use_spectral_norm=False,
    ):
        super(CoMBDBlock, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        self.convs = torch.nn.ModuleList()
        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(
                norm_f(
                    Conv1d(
                        in_channels=_f[0],
                        out_channels=_f[1],
                        kernel_size=_k,
                        stride=_s,
                        dilation=_d,
                        groups=_g,
                        padding=_p,
                    )
                )
            )
        self.projection_conv = norm_f(
            Conv1d(
                in_channels=filters[-1][1],
                out_channels=op_f,
                kernel_size=op_k,
                groups=op_g,
            )
        )

    def forward(self, x):
        """
        Calculate forward propagation.

        This method performs the forward pass of the AvocodoGenerator model,
        taking in an input tensor and an optional global conditioning tensor,
        and producing a list of output tensors.

        Args:
            c (Tensor): Input tensor of shape (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). If provided, it will be added to the
                input tensor after the initial convolution.

        Returns:
            List[Tensor]: List of output tensors, each of shape
                (B, out_channels, T).

        Examples:
            >>> generator = AvocodoGenerator()
            >>> input_tensor = torch.randn(1, 80, 160)  # Example input
            >>> global_tensor = torch.randn(1, 256, 1)  # Example global cond.
            >>> outputs = generator(input_tensor, global_tensor)
            >>> print([output.shape for output in outputs])
            [(1, 1, 40), (1, 1, 80)]  # Example output shapes

        Note:
            The output tensors will be generated at different scales based on
            the upsampling strategy defined in the generator's initialization.
        """
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


class CoMBD(torch.nn.Module):
    """
    CoMBD (Collaborative Multi-band Discriminator) module.

    This module implements the Collaborative Multi-band Discriminator as
    described in the paper: https://arxiv.org/abs/2206.13404. It processes
    input signals using a series of convolutional blocks, which can be
    configured for spectral normalization.

    Attributes:
        h (dict): Configuration parameters for the CoMBD module.
        pqmf (List[PQMF]): List of PQMF instances for signal analysis.

    Args:
        h (dict): A dictionary containing configuration parameters such as
            "combd_h_u", "combd_d_k", "combd_d_s", "combd_d_d", "combd_d_g",
            "combd_d_p", "combd_op_f", "combd_op_k", and "combd_op_g".
        pqmf_list (Optional[List[PQMF]]): List of PQMF instances. If None,
            default PQMF instances will be created based on `h`.
        use_spectral_norm (bool): Flag to determine whether to use spectral
            normalization in convolutional layers.

    Examples:
        >>> combd_params = {
        ...     "combd_h_u": [[16, 64, 256], [16, 64, 256]],
        ...     "combd_d_k": [[7, 11], [11, 21]],
        ...     "combd_d_s": [[1, 1], [1, 1]],
        ...     "combd_d_d": [[1, 1], [1, 1]],
        ...     "combd_d_g": [[1, 4], [1, 4]],
        ...     "combd_d_p": [[3, 5], [5, 10]],
        ...     "combd_op_f": [1, 1],
        ...     "combd_op_k": [3, 3],
        ...     "combd_op_g": [1, 1],
        ... }
        >>> combd = CoMBD(combd_params)
        >>> output_real, output_fake, fmaps_real, fmaps_fake = combd(ys, ys_hat)

    Returns:
        Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
        A tuple containing the output tensors for real and fake signals,
        along with the feature maps for each Conv1d layer for both real
        and fake signals.
    """

    def __init__(self, h, pqmf_list=None, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        self.h = h
        if pqmf_list is not None:
            self.pqmf = pqmf_list
        else:
            self.pqmf = [PQMF(*h.pqmf_config["lv2"]), PQMF(*h.pqmf_config["lv1"])]

        self.blocks = torch.nn.ModuleList()
        for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
            h["combd_h_u"],
            h["combd_d_k"],
            h["combd_d_s"],
            h["combd_d_d"],
            h["combd_d_g"],
            h["combd_d_p"],
            h["combd_op_f"],
            h["combd_op_k"],
            h["combd_op_g"],
        ):
            self.blocks.append(
                CoMBDBlock(
                    _h_u,
                    _d_k,
                    _d_s,
                    _d_d,
                    _d_g,
                    _d_p,
                    _op_f,
                    _op_k,
                    _op_g,
                )
            )

    def _block_forward(self, input, blocks, outs, f_maps):
        for x, block in zip(input, blocks):
            out, f_map = block(x)
            outs.append(out)
            f_maps.append(f_map)
        return outs, f_maps

    def _pqmf_forward(self, ys, ys_hat):
        # preprocess for multi_scale forward
        multi_scale_inputs = []
        multi_scale_inputs_hat = []
        for pqmf in self.pqmf:
            multi_scale_inputs.append(pqmf.to(ys[-1]).analysis(ys[-1])[:, :1, :])
            multi_scale_inputs_hat.append(
                pqmf.to(ys[-1]).analysis(ys_hat[-1])[:, :1, :]
            )

        outs_real = []
        f_maps_real = []
        # real
        # for hierarchical forward
        outs_real, f_maps_real = self._block_forward(
            ys, self.blocks, outs_real, f_maps_real
        )
        # for multi_scale forward
        outs_real, f_maps_real = self._block_forward(
            multi_scale_inputs, self.blocks[:-1], outs_real, f_maps_real
        )

        outs_fake = []
        f_maps_fake = []
        # predicted
        # for hierarchical forward
        outs_fake, f_maps_fake = self._block_forward(
            ys_hat, self.blocks, outs_fake, f_maps_fake
        )
        # for multi_scale forward
        outs_fake, f_maps_fake = self._block_forward(
            multi_scale_inputs_hat, self.blocks[:-1], outs_fake, f_maps_fake
        )

        return outs_real, outs_fake, f_maps_real, f_maps_fake

    def forward(self, ys, ys_hat):
        """
            Calculate forward propagation.

        This method performs the forward pass of the AvocodoGenerator, which
        takes an input tensor and optionally a global conditioning tensor.
        It applies a series of convolutional layers, upsampling, and residual
        blocks to generate the output tensors.

        Args:
            c (Tensor): Input tensor of shape (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). If provided, it is added to the
                input tensor after the initial convolution.

        Returns:
            List[Tensor]: List of output tensors, each of shape
            (B, out_channels, T).

        Examples:
            >>> generator = AvocodoGenerator()
            >>> input_tensor = torch.randn(8, 80, 100)  # Batch size 8, 80 channels, T=100
            >>> global_tensor = torch.randn(8, 256, 1)  # Global conditioning
            >>> outputs = generator(input_tensor, global_tensor)
            >>> for output in outputs:
            ...     print(output.shape)  # Should print shapes corresponding to output channels

        Note:
            The number of output tensors depends on the upsampling
            configuration of the generator. Each output tensor corresponds
            to a different stage of the upsampling process.
        """
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(ys, ys_hat)
        return outs_real, outs_fake, f_maps_real, f_maps_fake


# SBD
class MDC(torch.nn.Module):
    """
    Multiscale Dilated Convolution module.

    This class implements a multiscale dilated convolution block as described
    in the paper: https://arxiv.org/pdf/1609.07093.pdf. It utilizes dilated
    convolutions to increase the receptive field without increasing the number
    of parameters, making it suitable for various tasks such as time series
    prediction and audio processing.

    Attributes:
        d_convs (ModuleList): A list of dilated convolution layers.
        post_conv (Conv1d): A convolution layer applied after the dilated
            convolutions.
        softmax (Softmax): A softmax layer applied to the output.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        strides (int): Stride value for the post-convolution layer.
        kernel_size (List[int]): List of kernel sizes for the dilated
            convolutions.
        dilations (List[int]): List of dilation rates for the dilated
            convolutions.
        use_spectral_norm (bool): Whether to use spectral normalization for
            the convolution layers.

    Examples:
        >>> mdc = MDC(in_channels=64, out_channels=128, strides=1,
        ...            kernel_size=[3, 5], dilations=[1, 2])
        >>> input_tensor = torch.randn(10, 64, 100)  # (batch_size, channels, length)
        >>> output = mdc(input_tensor)
        >>> print(output.shape)
        torch.Size([10, 128, new_length])  # new_length depends on kernel_size
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False,
    ):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = torch.nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=_k,
                        dilation=_d,
                        padding=get_padding(_k, _d),
                    )
                )
            )
        self.post_conv = norm_f(
            Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=strides,
                padding=get_padding(_k, _d),
            )
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
            Calculate forward propagation.

        This method performs the forward pass through the Avocodo generator.
        It processes the input tensor `c` and optionally uses the global
        conditioning tensor `g`. The output is a list of tensors representing
        the generated outputs.

        Args:
            c (Tensor): Input tensor of shape (B, in_channels, T), where B is
                the batch size, in_channels is the number of input channels,
                and T is the length of the input sequence.
            g (Optional[Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). This is an optional input that can
                provide additional context for the generation.

        Returns:
            List[Tensor]: A list of output tensors, each of shape
                (B, out_channels, T), where out_channels is the number of
                output channels. The length of the list corresponds to the
                number of upsampling layers.

        Examples:
            >>> generator = AvocodoGenerator()
            >>> input_tensor = torch.randn(2, 80, 100)  # Batch size 2, 80 channels, 100 time steps
            >>> global_conditioning = torch.randn(2, 64, 1)  # Batch size 2, 64 global channels
            >>> outputs = generator(input_tensor, global_conditioning)
            >>> for output in outputs:
            ...     print(output.shape)
            torch.Size([2, 1, 100])
            torch.Size([2, 1, 50])
            torch.Size([2, 1, 25])
            torch.Size([2, 1, 12])

        Note:
            Ensure that the dimensions of the input tensor `c` and the
            global conditioning tensor `g` match the expected shapes.
        """
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):
    """
    SBD (Sub-band Discriminator) Block.

    This class implements a sub-band discriminator block, which applies
    multiple dilated convolutions to input audio segments to capture
    multi-resolution features. It is part of a larger framework designed
    for audio synthesis and evaluation.

    Attributes:
        convs (torch.nn.ModuleList): List of convolutional layers for
            processing input features.
        post_conv (torch.nn.Module): Final convolutional layer for output
            projection.

    Args:
        segment_dim (int): Dimension of the input segments.
        strides (List[int]): List of strides for each convolutional layer.
        filters (List[int]): List of output filter sizes for each layer.
        kernel_size (List[List[int]]): List of kernel sizes for each layer.
        dilations (List[List[int]]): List of dilation rates for each layer.
        use_spectral_norm (bool): Whether to apply spectral normalization
            to the convolutional layers.

    Returns:
        Tuple[Tensor, List[Tensor]]: A tuple containing the output tensor
        of shape (B, C_out, T_out) and a list of feature maps of shape
        (B, C, T) at each Conv1d layer.

    Examples:
        >>> sbd_block = SBDBlock(segment_dim=64, strides=[1, 1],
        ...                       filters=[32, 64],
        ...                       kernel_size=[[3, 3], [3, 3]],
        ...                       dilations=[[1, 2], [1, 2]])
        >>> input_tensor = torch.randn(10, 64, 128)  # (B, C_in, T_in)
        >>> output, feature_maps = sbd_block(input_tensor)
        >>> print(output.shape)  # Should output (10, 1, T_out)
        >>> print(len(feature_maps))  # Should match number of conv layers

    Note:
        The output shape (B, C_out, T_out) will depend on the input
        dimensions, the strides, and the kernel sizes used in the
        convolutional layers.

    Raises:
        ValueError: If the input dimensions do not match expected shapes
        for the convolutional layers.
    """

    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
        use_spectral_norm=False,
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = torch.nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])
        for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size, dilations):
            self.convs.append(
                MDC(
                    in_channels=_f[0],
                    out_channels=_f[1],
                    strides=_s,
                    kernel_size=_k,
                    dilations=_d,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        self.post_conv = norm_f(
            Conv1d(
                in_channels=_f[1],
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
            )
        )  # @@

    def forward(self, x):
        """
        Calculate forward propagation.

        This method performs the forward pass of the AvocodoGenerator model.
        It takes an input tensor `c` and an optional global conditioning tensor `g`,
        processes them through the defined layers, and returns the output tensor.

        Args:
            c (Tensor): Input tensor of shape (B, in_channels, T), where B is the
                batch size, in_channels is the number of input channels, and T is
                the length of the input sequence.
            g (Optional[Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). If provided, it is added to the output of
                the initial convolution layer.

        Returns:
            List[Tensor]: List of output tensors, each of shape (B, out_channels, T),
                where each tensor corresponds to an upsampling stage in the generator.

        Examples:
            >>> generator = AvocodoGenerator()
            >>> input_tensor = torch.randn(8, 80, 100)  # Batch of 8, 80 channels, 100 time steps
            >>> output = generator(input_tensor)
            >>> len(output)  # Number of outputs corresponds to the number of upsample stages
            4

        Note:
            The output tensors from the generator correspond to different scales
            of the generated audio signal.
        """
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)  # @@

        return x, fmap


class MDCDConfig:
    """
    Configuration class for the Multi-band Discriminator (MDC).

    This class holds the configuration parameters required for the
    Multi-band Discriminator, including PQMF parameters, filter sizes,
    kernel sizes, dilations, strides, band ranges, and segment size.

    Attributes:
        pqmf_params (List[int]): Parameters for PQMF configuration.
        f_pqmf_params (List[int]): Parameters for filtered PQMF configuration.
        filters (List[List[int]]): List of filter sizes for the discriminator.
        kernel_sizes (List[List[int]]): List of kernel sizes for the
            convolutional layers.
        dilations (List[List[int]]): List of dilations for the
            convolutional layers.
        strides (List[List[int]]): List of strides for the convolutional layers.
        band_ranges (List[List[int]]): List of ranges for each frequency band.
        transpose (List[bool]): Indicates whether to transpose the input
            for each band.
        segment_size (int): Size of segments to be processed.

    Args:
        h (Dict[str, Any]): A dictionary containing configuration parameters.

    Examples:
        >>> config = MDCDConfig({
        ...     "pqmf_config": {"sbd": [16, 256, 0.03, 10.0],
        ...                     "fsbd": [64, 256, 0.1, 9.0]},
        ...     "sbd_filters": [[64, 128, 256], [32, 64, 128]],
        ...     "sbd_kernel_sizes": [[[3, 3, 3], [5, 5, 5]]],
        ...     "sbd_dilations": [[[1, 2, 3], [1, 2, 3]]],
        ...     "sbd_strides": [[1, 1, 1]],
        ...     "sbd_band_ranges": [[0, 16]],
        ...     "sbd_transpose": [False],
        ...     "segment_size": 8192
        ... })
        >>> print(config.filters)
        [[64, 128, 256], [32, 64, 128]]
    """

    def __init__(self, h):
        self.pqmf_params = h["pqmf_config"]["sbd"]
        self.f_pqmf_params = h["pqmf_config"]["fsbd"]
        self.filters = h["sbd_filters"]
        self.kernel_sizes = h["sbd_kernel_sizes"]
        self.dilations = h["sbd_dilations"]
        self.strides = h["sbd_strides"]
        self.band_ranges = h["sbd_band_ranges"]
        self.transpose = h["sbd_transpose"]
        self.segment_size = h["segment_size"]


class SBD(torch.nn.Module):
    """
    SBD (Sub-band Discriminator) from https://arxiv.org/pdf/2206.13404.pdf

    This module implements a Sub-band Discriminator designed for audio
    processing tasks, utilizing multi-band analysis to discriminate
    between real and generated audio signals. It processes the input
    signals using a series of convolutional blocks, which are structured
    to capture the characteristics of different frequency bands.

    Attributes:
        config (MDCDConfig): Configuration parameters for the SBD.
        pqmf (PQMF): Perfect Reconstruction Filter Bank for signal analysis.
        f_pqmf (PQMF or None): Secondary PQMF for frequency analysis, if needed.
        discriminators (ModuleList): List of sub-band discriminators.

    Args:
        h (Dict[str, Any]): Configuration dictionary containing filter,
            kernel sizes, dilations, strides, band ranges, and transpose
            parameters for the discriminators.
        use_spectral_norm (bool): Flag to indicate whether to use spectral
            normalization for the convolutional layers.

    Returns:
        None

    Examples:
        >>> sbd_config = {
        ...     "sbd_filters": [[64, 128], [128, 256]],
        ...     "sbd_kernel_sizes": [[[3, 3], [3, 3]]],
        ...     "sbd_dilations": [[[1, 2], [1, 2]]],
        ...     "sbd_strides": [[1, 1]],
        ...     "sbd_band_ranges": [[0, 1]],
        ...     "sbd_transpose": [False],
        ...     "pqmf_config": {"sbd": [16, 256, 0.03, 10.0]}
        ... }
        >>> sbd = SBD(sbd_config)
        >>> real_audio = torch.randn(1, 1, 8192)
        >>> fake_audio = torch.randn(1, 1, 8192)
        >>> real_outputs, fake_outputs, real_fmaps, fake_fmaps = sbd(real_audio, fake_audio)

    Note:
        The SBD is part of a larger framework designed for GAN-based
        audio synthesis, and it is specifically tailored for evaluating
        the quality of generated audio signals.
    """

    def __init__(self, h, use_spectral_norm=False):
        super(SBD, self).__init__()
        self.config = MDCDConfig(h)
        self.pqmf = PQMF(*self.config.pqmf_params)
        if True in h["sbd_transpose"]:
            self.f_pqmf = PQMF(*self.config.f_pqmf_params)
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose,
        ):
            if _tr:
                segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            self.discriminators.append(
                SBDBlock(
                    segment_dim=segment_dim,
                    filters=_f,
                    kernel_size=_k,
                    dilations=_d,
                    strides=_s,
                    use_spectral_norm=use_spectral_norm,
                )
            )

    def forward(self, y, y_hat):
        """
            Calculate forward propagation.

        This method performs the forward pass of the AvocodoGenerator module.
        It takes an input tensor and, optionally, a global conditioning tensor,
        and computes the output tensors through a series of convolutional layers.

        Args:
            c (Tensor): Input tensor of shape (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor of shape
                (B, global_channels, 1). If provided, it is added to the input
                tensor after the initial convolution.

        Returns:
            List[Tensor]: A list of output tensors, each of shape
                (B, out_channels, T), representing the generated outputs.

        Examples:
            >>> generator = AvocodoGenerator()
            >>> input_tensor = torch.randn(1, 80, 100)  # (B, in_channels, T)
            >>> global_tensor = torch.randn(1, 256, 1)  # (B, global_channels, 1)
            >>> output = generator(input_tensor, global_tensor)
            >>> len(output)  # Output will be a list of tensors
            4

        Note:
            The input tensor should have the specified number of input channels
            and the global conditioning tensor should have the specified number
            of global channels if provided.
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(
            self.discriminators, self.config.band_ranges, self.config.transpose
        ):
            if tr:
                _y_in = y_in_f[:, br[0] : br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0] : br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            else:
                _y_in = y_in[:, br[0] : br[1], :]
                _y_hat_in = y_hat_in[:, br[0] : br[1], :]
            y_d_r, fmap_r = d(_y_in)
            y_d_g, fmap_g = d(_y_hat_in)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class AvocodoDiscriminator(torch.nn.Module):
    """
    Avocodo Discriminator module.

    This class implements the Avocodo Discriminator, which combines
    Collaborative Multi-band Discriminator (CoMBD) and Sub-band
    Discriminator (SBD) for enhanced performance in generative
    adversarial networks. It processes input signals and predicts
    whether they are real or fake.

    Attributes:
        pqmf_lv2 (PQMF): PQMF for level 2 processing.
        pqmf_lv1 (PQMF): PQMF for level 1 processing.
        combd (CoMBD): Collaborative Multi-band Discriminator instance.
        sbd (SBD): Sub-band Discriminator instance.
        projection_filters (List[int]): Filters for projection layers.

    Args:
        combd (Dict[str, Any]): Configuration dictionary for CoMBD.
        sbd (Dict[str, Any]): Configuration dictionary for SBD.
        pqmf_config (Dict[str, Any]): Configuration dictionary for PQMF.
        projection_filters (List[int]): List of projection filters for the
            output.

    Examples:
        >>> discriminator = AvocodoDiscriminator()
        >>> real_signal = torch.randn(1, 1, 8192)  # Batch of real signal
        >>> fake_signal = torch.randn(1, 1, 8192)  # Batch of fake signal
        >>> outs_real, outs_fake, fmaps_real, fmaps_fake = discriminator(
        ...     real_signal, fake_signal
        ... )

    Returns:
        List[List[torch.Tensor]]: Outputs containing real and fake
        predictions along with feature maps for both.

    Note:
        The discriminator uses spectral normalization if specified in the
        configuration.
    """

    def __init__(
        self,
        combd: Dict[str, Any] = {
            "combd_h_u": [
                [16, 64, 256, 1024, 1024, 1024],
                [16, 64, 256, 1024, 1024, 1024],
                [16, 64, 256, 1024, 1024, 1024],
            ],
            "combd_d_k": [
                [7, 11, 11, 11, 11, 5],
                [11, 21, 21, 21, 21, 5],
                [15, 41, 41, 41, 41, 5],
            ],
            "combd_d_s": [
                [1, 1, 4, 4, 4, 1],
                [1, 1, 4, 4, 4, 1],
                [1, 1, 4, 4, 4, 1],
            ],
            "combd_d_d": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            "combd_d_g": [
                [1, 4, 16, 64, 256, 1],
                [1, 4, 16, 64, 256, 1],
                [1, 4, 16, 64, 256, 1],
            ],
            "combd_d_p": [
                [3, 5, 5, 5, 5, 2],
                [5, 10, 10, 10, 10, 2],
                [7, 20, 20, 20, 20, 2],
            ],
            "combd_op_f": [1, 1, 1],
            "combd_op_k": [3, 3, 3],
            "combd_op_g": [1, 1, 1],
        },
        sbd: Dict[str, Any] = {
            "use_sbd": True,
            "sbd_filters": [
                [64, 128, 256, 256, 256],
                [64, 128, 256, 256, 256],
                [64, 128, 256, 256, 256],
                [32, 64, 128, 128, 128],
            ],
            "sbd_strides": [
                [1, 1, 3, 3, 1],
                [1, 1, 3, 3, 1],
                [1, 1, 3, 3, 1],
                [1, 1, 3, 3, 1],
            ],
            "sbd_kernel_sizes": [
                [[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
                [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
                [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
            ],
            "sbd_dilations": [
                [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
                [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]],
            ],
            "sbd_band_ranges": [[0, 6], [0, 11], [0, 16], [0, 64]],
            "sbd_transpose": [False, False, False, True],
            "pqmf_config": {
                "sbd": [16, 256, 0.03, 10.0],
                "fsbd": [64, 256, 0.1, 9.0],
            },
            "segment_size": 8192,
        },
        pqmf_config: Dict[str, Any] = {
            "lv1": [2, 256, 0.25, 10.0],
            "lv2": [4, 192, 0.13, 10.0],
        },
        projection_filters: List[int] = [0, 1, 1, 1],
    ):
        super(AvocodoDiscriminator, self).__init__()

        self.pqmf_lv2 = PQMF(*pqmf_config["lv2"])
        self.pqmf_lv1 = PQMF(*pqmf_config["lv1"])
        self.combd = CoMBD(
            combd,
            [self.pqmf_lv2, self.pqmf_lv1],
            use_spectral_norm=combd["use_spectral_norm"],
        )
        self.sbd = SBD(
            sbd,
            use_spectral_norm=sbd["use_spectral_norm"],
        )
        self.projection_filters = projection_filters

    def forward(
        self, y: torch.Tensor, y_hats: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """
        Perform forward propagation through the Avocodo Discriminator.

        This method computes the outputs of the discriminator given the real
        and generated signals. It processes the inputs through multiple
        layers and combines the outputs from different discriminators.

        Args:
            y (torch.Tensor): The real signals of shape (B, C, T).
            y_hats (torch.Tensor): The generated signals of shape (B, C, T).

        Returns:
            List[List[torch.Tensor]]: A list containing the outputs of the
            discriminators. Specifically, it returns:
            - outs_real: List of output tensors for real signals.
            - outs_fake: List of output tensors for generated signals.
            - fmaps_real: List of feature maps for real signals.
            - fmaps_fake: List of feature maps for generated signals.

        Examples:
            >>> discriminator = AvocodoDiscriminator(...)
            >>> real_signals = torch.randn(8, 1, 256)  # Batch of real signals
            >>> generated_signals = torch.randn(8, 1, 256)  # Batch of generated signals
            >>> outputs_real, outputs_fake, fmap_real, fmap_fake = discriminator(real_signals, generated_signals)

        Note:
            The forward method expects the input tensors to have a specific
            shape, where B is the batch size, C is the number of channels,
            and T is the time dimension.
        """
        ys = [
            self.pqmf_lv2.analysis(y)[:, : self.projection_filters[1]],
            self.pqmf_lv1.analysis(y)[:, : self.projection_filters[2]],
            y,
        ]

        (
            combd_outs_real,
            combd_outs_fake,
            combd_fmaps_real,
            combd_fmaps_fake,
        ) = self.combd(ys, y_hats)

        sbd_outs_real, sbd_outs_fake, sbd_fmaps_real, sbd_fmaps_fake = self.sbd(
            y, y_hats[-1]
        )

        # Combine the outputs of both discriminators
        outs_real = combd_outs_real + sbd_outs_real
        outs_fake = combd_outs_fake + sbd_outs_fake
        fmaps_real = combd_fmaps_real + sbd_fmaps_real
        fmaps_fake = combd_fmaps_fake + sbd_fmaps_fake

        return outs_real, outs_fake, fmaps_real, fmaps_fake


class AvocodoDiscriminatorPlus(torch.nn.Module):
    """
    Avocodo discriminator with additional multi-frequency discriminator.

    This class extends the Avocodo Discriminator by incorporating a
    Multi-Frequency Discriminator (MFD) for enhanced feature extraction
    from audio signals. It combines outputs from the Collaborative
    Multi-band Discriminator (CoMBD), Sub-band Discriminator (SBD),
    and the MFD to produce a more comprehensive analysis of real
    and generated audio data.

    Attributes:
        pqmf_lv2 (PQMF): PQMF object for level 2 processing.
        pqmf_lv1 (PQMF): PQMF object for level 1 processing.
        combd (CoMBD): Instance of the Collaborative Multi-band Discriminator.
        sbd (SBD): Instance of the Sub-band Discriminator.
        mfd (MultiFrequencyDiscriminator): Instance of the Multi-Frequency Discriminator.
        projection_filters (List[int]): Filters for the projection layers.

    Args:
        combd (Dict[str, Any]): Configuration parameters for CoMBD.
        sbd (Dict[str, Any]): Configuration parameters for SBD.
        pqmf_config (Dict[str, Any]): Configuration for PQMF.
        projection_filters (List[int]): Projection filters for output layers.
        sample_rate (int): Sample rate of the audio signals.
        multi_freq_disc_params (Dict[str, Any]): Parameters for MFD.

    Returns:
        List[List[torch.Tensor]]: A list containing outputs and feature maps
                                   from the discriminators.

    Examples:
        >>> discriminator = AvocodoDiscriminatorPlus()
        >>> real_audio = torch.randn(1, 1, 8192)  # Example real audio tensor
        >>> fake_audio = torch.randn(1, 1, 8192)  # Example generated audio tensor
        >>> outputs_real, outputs_fake, fmaps_real, fmaps_fake = discriminator(real_audio, fake_audio)

    Note:
        The class utilizes the `torch` library for neural network
        functionalities and requires the input tensors to be in the
        shape of (B, C, T), where B is the batch size, C is the
        number of channels, and T is the length of the time series.

    Todo:
        - Implement additional methods for enhanced functionality.
    """

    def __init__(
        self,
        combd: Dict[str, Any] = {
            "combd_h_u": [
                [16, 64, 256, 1024, 1024, 1024],
                [16, 64, 256, 1024, 1024, 1024],
                [16, 64, 256, 1024, 1024, 1024],
            ],
            "combd_d_k": [
                [7, 11, 11, 11, 11, 5],
                [11, 21, 21, 21, 21, 5],
                [15, 41, 41, 41, 41, 5],
            ],
            "combd_d_s": [
                [1, 1, 4, 4, 4, 1],
                [1, 1, 4, 4, 4, 1],
                [1, 1, 4, 4, 4, 1],
            ],
            "combd_d_d": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            "combd_d_g": [
                [1, 4, 16, 64, 256, 1],
                [1, 4, 16, 64, 256, 1],
                [1, 4, 16, 64, 256, 1],
            ],
            "combd_d_p": [
                [3, 5, 5, 5, 5, 2],
                [5, 10, 10, 10, 10, 2],
                [7, 20, 20, 20, 20, 2],
            ],
            "combd_op_f": [1, 1, 1],
            "combd_op_k": [3, 3, 3],
            "combd_op_g": [1, 1, 1],
        },
        sbd: Dict[str, Any] = {
            "use_sbd": True,
            "sbd_filters": [
                [64, 128, 256, 256, 256],
                [64, 128, 256, 256, 256],
                [64, 128, 256, 256, 256],
                [32, 64, 128, 128, 128],
            ],
            "sbd_strides": [
                [1, 1, 3, 3, 1],
                [1, 1, 3, 3, 1],
                [1, 1, 3, 3, 1],
                [1, 1, 3, 3, 1],
            ],
            "sbd_kernel_sizes": [
                [[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
                [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
                [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
            ],
            "sbd_dilations": [
                [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
                [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]],
            ],
            "sbd_band_ranges": [[0, 6], [0, 11], [0, 16], [0, 64]],
            "sbd_transpose": [False, False, False, True],
            "pqmf_config": {
                "sbd": [16, 256, 0.03, 10.0],
                "fsbd": [64, 256, 0.1, 9.0],
            },
            "segment_size": 8192,
        },
        pqmf_config: Dict[str, Any] = {
            "lv1": [2, 256, 0.25, 10.0],
            "lv2": [4, 192, 0.13, 10.0],
        },
        projection_filters: List[int] = [0, 1, 1, 1],
        # Multi-frequency discriminator related
        sample_rate: int = 22050,
        multi_freq_disc_params: Dict[str, Any] = {
            "hop_length_factors": [4, 8, 16],
            "hidden_channels": [256, 512, 512],
            "domain": "double",
            "mel_scale": True,
            "divisors": [32, 16, 8, 4, 2, 1, 1],
            "strides": [1, 2, 1, 2, 1, 2, 1],
        },
    ):
        super().__init__()

        self.pqmf_lv2 = PQMF(*pqmf_config["lv2"])
        self.pqmf_lv1 = PQMF(*pqmf_config["lv1"])
        self.combd = CoMBD(
            combd,
            [self.pqmf_lv2, self.pqmf_lv1],
            use_spectral_norm=combd["use_spectral_norm"],
        )
        self.sbd = SBD(
            sbd,
            use_spectral_norm=sbd["use_spectral_norm"],
        )
        # Multi-frequency discriminator related
        if "hop_lengths" not in multi_freq_disc_params:
            # Transfer hop lengths factors to hop lengths
            multi_freq_disc_params["hop_lengths"] = []

            for i in range(len(multi_freq_disc_params["hop_length_factors"])):
                multi_freq_disc_params["hop_lengths"].append(
                    int(
                        sample_rate
                        * multi_freq_disc_params["hop_length_factors"][i]
                        / 1000
                    )
                )

            del multi_freq_disc_params["hop_length_factors"]

        self.mfd = MultiFrequencyDiscriminator(
            **multi_freq_disc_params,
        )
        self.projection_filters = projection_filters

    def forward(
        self, y: torch.Tensor, y_hats: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """
            Perform forward propagation through the AvocodoDiscriminatorPlus.

        This method takes input tensors and produces outputs from the
        combined multi-band and sub-band discriminators, as well as a
        multi-frequency discriminator. It analyzes both real and fake
        signals to generate corresponding output tensors and feature maps.

        Args:
            y (torch.Tensor): Ground truth signal tensor of shape
                              (B, C, T).
            y_hats (torch.Tensor): Predicted signal tensor of shape
                                   (B, C, T).

        Returns:
            List[List[torch.Tensor]]: A list containing:
                - outs_real (List[Tensor]): List of output tensors for
                  real signals.
                - outs_fake (List[Tensor]): List of output tensors for
                  fake signals.
                - fmaps_real (List[List[Tensor]]): List of feature maps
                  for real signals at each layer.
                - fmaps_fake (List[List[Tensor]]): List of feature maps
                  for fake signals at each layer.

        Examples:
            >>> discriminator = AvocodoDiscriminatorPlus()
            >>> real_signal = torch.randn(1, 1, 1024)  # Example real signal
            >>> fake_signal = torch.randn(1, 1, 1024)  # Example fake signal
            >>> outs_real, outs_fake, fmaps_real, fmaps_fake = discriminator(real_signal, fake_signal)

        Note:
            The output tensors are produced by analyzing the input signals
            through multiple discriminators, which allows for more
            comprehensive assessment of the audio signals.
        """
        ys = [
            self.pqmf_lv2.analysis(y)[:, : self.projection_filters[1]],
            self.pqmf_lv1.analysis(y)[:, : self.projection_filters[2]],
            y,
        ]

        (
            combd_outs_real,
            combd_outs_fake,
            combd_fmaps_real,
            combd_fmaps_fake,
        ) = self.combd(ys, y_hats)

        sbd_outs_real, sbd_outs_fake, sbd_fmaps_real, sbd_fmaps_fake = self.sbd(
            y, y_hats[-1]
        )

        mfd_fmaps_real = self.mfd(y)
        mfd_fmaps_fake = self.mfd(y_hats[-1])
        mfd_outs_real = mfd_fmaps_real[-1]
        mfd_outs_fake = mfd_fmaps_fake[-1]

        # Combine the outputs of both discriminators
        outs_real = combd_outs_real + sbd_outs_real + mfd_outs_real
        outs_fake = combd_outs_fake + sbd_outs_fake + mfd_outs_fake
        fmaps_real = combd_fmaps_real + sbd_fmaps_real + mfd_fmaps_real
        fmaps_fake = combd_fmaps_fake + sbd_fmaps_fake + mfd_fmaps_fake

        return outs_real, outs_fake, fmaps_real, fmaps_fake
