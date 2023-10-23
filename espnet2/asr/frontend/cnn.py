# from https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py

import copy
import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend

class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
        conv_mode: str
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm

        if conv_mode == "standard":
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
        elif conv_mode == "depth_only":
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                groups=in_channels
            )
        elif conv_mode == "depth_sep":
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    groups=in_channels
                ),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias,
                )
            )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            # When input length is 0, the resulting length can be negative. So fix it here.
            length = torch.max(torch.zeros_like(length), length)
        return x, length

class CNNFrontend(AbsFrontend):
    def __init__(
        self,
        norm_mode: str,
        conv_mode: str,
        shapes: List[Tuple[int, int, int]],
        bias: bool,
    ):

        if norm_mode not in ["group_norm", "layer_norm"]:
            raise ValueError("Invalid norm mode")

        if conv_mode not in ["standard", "depth_only", "depth_sep"]:
            raise ValueError("Invalid cnn mode")

        locks = []
        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(shapes):
            normalization = None
            if norm_mode == "group_norm" and i == 0:
                normalization = nn.GroupNorm(
                    num_groups=out_channels,
                    num_channels=out_channels,
                    affine=True,
                )
            elif norm_mode == "layer_norm":
                normalization = LayerNorm(
                    normalized_shape=out_channels,
                    elementwise_affine=True,
                )
            blocks.append(
                ConvLayerBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    layer_norm=normalization,
                    conv_mode=conv_mode
                )
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*blocks)

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected the input Tensor to be 2D (batch, time). Found: {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x, length

    