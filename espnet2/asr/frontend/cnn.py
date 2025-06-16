# Adapted from TorchAudio
# github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py
from typing import List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F

from espnet2.asr.frontend.abs_frontend import AbsFrontend


def dim_1_layer_norm(x, eps=1e-05, gamma=None, beta=None):
    """Functional version of Dim1LayerNorm."""

    B, D, T = x.shape
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + eps)

    if gamma is not None:
        x = x * gamma.view(1, -1, 1)
        if beta is not None:
            x = x + beta.view(1, -1, 1)
    return x


class Dim1LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True):
        """LayerNorm on middle dim.

        It assumes the input is shape B, D, T
        to avoid transposing.
        Faster than TransposedLayerNorm, but
        may lead to minor numerical differences.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        assert x.size(1) == self.normalized_shape
        return dim_1_layer_norm(x, self.eps, self.weight, self.bias)


class TransposedLayerNorm(nn.LayerNorm):
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
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """ConvLayerBlock Forward.

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
        normalize_output: bool = False,
        layer_norm_cls: Literal["transposed", "dim1"] = "transposed",
    ):

        super().__init__()

        if norm_mode not in ["group_norm", "layer_norm"]:
            raise ValueError("Invalid norm mode")

        if conv_mode not in ["standard", "depth_only", "depth_sep"]:
            raise ValueError("Invalid cnn mode")

        self.output_channels = shapes[-1][0]
        self.normalize_audio = normalize_audio

        if layer_norm_cls == "dim1":
            layer_norm_func = Dim1LayerNorm
        else:
            layer_norm_func = TransposedLayerNorm

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
                normalization = layer_norm_func(
                    normalized_shape=out_channels,
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

        if normalize_output:
            self.final_norm = nn.LayerNorm(self.output_channels)
        else:
            self.final_norm = nn.Identity()

    def output_size(self) -> int:
        return self.output_channels

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """CNNFrontend Forward.

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
        x = self.final_norm(x)
        return x, length
