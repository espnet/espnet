import argparse
import logging
from typing import Dict, Optional

import torch
from typeguard import typechecked

from espnet2.uasr.generator.abs_generator import AbsGenerator
from espnet2.utils.types import str2bool


class TransposeLast(torch.nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvGenerator(AbsGenerator):
    """convolutional generator for UASR."""

    @typechecked
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: Optional[Dict] = None,
        conv_kernel: int = 3,
        conv_dilation: int = 1,
        conv_stride: int = 9,
        pad: int = -1,
        bias: str2bool = False,
        dropout: float = 0.0,
        batch_norm: str2bool = True,
        batch_norm_weight: float = 30.0,
        residual: str2bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if cfg is not None:
            cfg = argparse.Namespace(**cfg)
            self.conv_kernel = cfg.generator_kernel
            self.conv_dilation = cfg.generator_dilation
            self.conv_stride = cfg.generator_stride
            self.pad = cfg.generator_pad
            self.bias = cfg.generator_bias
            self.dropout = torch.nn.Dropout(cfg.generator_dropout)
            # TODO(Dongji): batch_norm is not in cfg
            self.batch_norm = False
            self.batch_norm_weight = cfg.generator_batch_norm
            self.residual = cfg.generator_residual
        else:
            self.conv_kernel = conv_kernel
            self.conv_dilation = conv_dilation
            self.conv_stride = conv_stride
            self.output_dim = output_dim
            self.pad = pad
            self.bias = bias
            self.dropout = torch.nn.Dropout(dropout)
            self.batch_norm = batch_norm
            self.batch_norm_weight = batch_norm_weight
            self.residual = residual

        if self.pad < 0:
            self.padding = self.conv_kernel // 2
        else:
            self.padding = self.pad

        self.proj = torch.nn.Sequential(
            TransposeLast(),
            torch.nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                dilation=self.conv_dilation,
                padding=self.padding,
                bias=self.bias,
            ),
            TransposeLast(),
        )
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(self.batch_norm_weight)
        if self.residual:
            self.in_proj = torch.nn.Linear(input_dim, input_dim)

    def output_size(self):
        return self.output_dim

    def forward(
        self,
        feats: torch.Tensor,
        text: Optional[torch.Tensor],
        feats_padding_mask: torch.Tensor,
    ):
        inter_x = None
        if self.batch_norm:
            feats = self.bn_padded_data(feats, feats_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(feats))
            feats = feats + inter_x

        feats = self.dropout(feats)

        generated_sample = self.proj(feats)
        generated_sample_padding_mask = feats_padding_mask[:, :: self.conv_stride]

        if generated_sample_padding_mask.size(1) != generated_sample.size(1):
            new_padding = generated_sample_padding_mask.new_zeros(
                generated_sample.shape[:-1]
            )
            diff = new_padding.size(1) - generated_sample_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = generated_sample_padding_mask
            else:
                logging.info("ATTENTION: make sure that you are using V2 instead of V1")
                assert diff < 0
                new_padding = generated_sample_padding_mask[:, :diff]

            generated_sample_padding_mask = new_padding

        real_sample = None
        if text is not None:
            assert torch.count_nonzero(text) > 0
            real_sample = generated_sample.new_zeros(text.numel(), self.output_dim)
            real_sample.scatter_(1, text.view(-1, 1).long(), 1)
            real_sample = real_sample.view(text.shape + (self.output_dim,))

        return generated_sample, real_sample, inter_x, generated_sample_padding_mask

    def bn_padded_data(self, feature: torch.Tensor, padding_mask: torch.Tensor):
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature
