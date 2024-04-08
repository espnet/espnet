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
    return int((kernel_size * dilation - dilation) / 2)


class AvocodoGenerator(torch.nn.Module):
    """Avocodo generator module."""

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
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            List[Tensor]: List of output tensors (B, out_channels, T).

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
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


# CoMBD
class CoMBDBlock(torch.nn.Module):
    """CoMBD (Collaborative Multi-band Discriminator) block module"""

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
        """Forward pass through the CoMBD block.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, T_in).

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the output tensor of
                                         shape (B, C_out, T_out)
            and a list of feature maps of shape (B, C, T) at each Conv1d layer.
        """
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


class CoMBD(torch.nn.Module):
    """CoMBD (Collaborative Multi-band Discriminator) module

    from from https://arxiv.org/abs/2206.13404
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
        """Forward CoMBD.

        Args:
            ys (List[Tensor]): List of ground truth signals of shape (B, 1, T).
            ys_hat (List[Tensor]): List of predicted signals of shape (B, 1, T).
        Returns:
            Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
            Tuple containing the list of output tensors of shape (B, C_out, T_out)
            for real and fake, respectively, and the list of feature maps of shape
            (B, C, T) at each Conv1d layer for real and fake, respectively.
        """
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(ys, ys_hat)
        return outs_real, outs_fake, f_maps_real, f_maps_fake


# SBD
class MDC(torch.nn.Module):
    """Multiscale Dilated Convolution from https://arxiv.org/pdf/1609.07093.pdf"""

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
    """SBD (Sub-band Discriminator) Block"""

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
        fmap = []
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)
        x = self.post_conv(x)  # @@

        return x, fmap


class MDCDConfig:
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
    """SBD (Sub-band Discriminator) from https://arxiv.org/pdf/2206.13404.pdf"""

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
    """Avocodo Discriminator module"""

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
    """Avocodo discriminator with additional MFD."""

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
