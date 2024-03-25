# Adapted from TorchAudio
# github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py

import copy
import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.transpose(-2, -1)
        return x


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
        conv_mode: str,
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
                groups=in_channels,
            )
        elif conv_mode == "depth_sep":
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    groups=in_channels,
                ),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias,
                ),
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
            x = torch.utils.checkpoint.checkpoint(
                self.layer_norm, x, use_reentrant=False
            )
        x = nn.functional.gelu(x)

        if length is not None:
            length = (
                torch.div(length - self.kernel_size, self.stride, rounding_mode="floor")
                + 1
            )
            # When input length is 0, the resulting length can be negative.
            length = torch.max(torch.zeros_like(length), length)
        return x, length


class CNNFrontend(AbsFrontend):
    """Convolutional feature extractor.

    Typically used in SSL models.
    Uses raw waveforms as input.
    """

    def __init__(
        self,
        norm_mode: str,
        conv_mode: str,
        bias: bool,
        shapes: List[Tuple[int, int, int]] = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        fs: Union[int, str] = 16000,
        normalize_audio: bool = False,
    ):

        super().__init__()

        if norm_mode not in ["group_norm", "layer_norm"]:
            raise ValueError("Invalid norm mode")

        if conv_mode not in ["standard", "depth_only", "depth_sep"]:
            raise ValueError("Invalid cnn mode")

        self.output_channels = shapes[-1][0]
        self.normalize_audio = normalize_audio

        blocks = []
        in_channels = 1
        self.downsampling_factor = 1
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
                    conv_mode=conv_mode,
                )
            )
            in_channels = out_channels
            self.downsampling_factor *= stride
        self.layers = nn.Sequential(*blocks)

    def output_size(self) -> int:
        return self.output_channels

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
            raise ValueError(
                f"Expected the input to be 2D (batch, time). Found: {list(x.shape)}"
            )

        if self.normalize_audio:
            x = F.layer_norm(x, x.shape)

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x, length
