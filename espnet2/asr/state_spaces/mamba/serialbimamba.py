# Copyright (c) 2023, Tri Dao, Albert Gu.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code is modified from the original script inspired by Mamba-ND [1]
#
# [1] Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data
#     https://arxiv.org/abs/2402.05892

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from espnet2.asr.state_spaces.mamba.ops.triton.layer_norm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class SerialBiMambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Partially Bidirectional Mamba Block similar to the one used in the "Mamba-ND" paper.

        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).


        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer_forward = mixer_cls(dim, prefix_bidir=True)
        self.mixer_backward = mixer_cls(dim, prefix_bidir=True)
        self.norm_f = norm_cls(dim)
        self.norm_b = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm_f, (nn.LayerNorm, RMSNorm)), (
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        inference_params=None,
        flip_fn=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required). (B, L, D)
            residual: hidden_states = Mixer(LN(residual))
            mask: the mask for the input sequence (optional).
        """
        hidden_states_f, residual = self.add_norm(hidden_states, residual, self.norm_f)
        hidden_states_f = self.mixer_forward(
            hidden_states_f, inference_params=inference_params, direction="forward"
        )
        hidden_states_f = hidden_states_f + residual
        if flip_fn is None:
            hidden_states_b, residual = self.add_norm(
                hidden_states_f, None, self.norm_b
            )
            hidden_states_b = self.mixer_backward(
                hidden_states_b, inference_params=inference_params, direction="backward"
            )
            hidden_states_b = hidden_states_b + residual
        else:
            hidden_states_b = flip_fn(hidden_states_f.transpose(-1, -2)).transpose(
                -1, -2
            )
            hidden_states_b, residual = self.add_norm(
                hidden_states_b, None, self.norm_b
            )
            hidden_states_b = self.mixer_backward(
                hidden_states_b, inference_params=inference_params, direction="backward"
            )
            hidden_states_b = hidden_states_b + residual
            hidden_states_b = flip_fn(hidden_states_b.transpose(-1, -2)).transpose(
                -1, -2
            )

        hidden_states = hidden_states_b
        residual = None
        return hidden_states, residual

    def add_norm(self, hidden_states, residual, norm):
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = norm(residual.to(dtype=norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                norm.weight,
                norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=norm.eps,
            )
        return hidden_states, residual

    def allocate_inference_cache(
        self, batch_size, max_seqlen=None, dtype=None, **kwargs
    ):
        conv_state_f, ssm_state_f = self.mixer_forward.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
        conv_state_b, ssm_state_b = self.mixer_backward.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
        return conv_state_f, ssm_state_f, conv_state_b, ssm_state_b
