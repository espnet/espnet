# -*- coding: utf-8 -*-

# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Style encoder of GST-Tacotron."""

from typing import List

import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class StyleTokenEncoder(torch.nn.Module):
    """Style token encoder."""

    def __init__(
        self,
        num_mels: int = 80,
        num_tokens: int = 10,
        token_dim: int = 256,
        num_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: List[int] = [32, 32, 64, 64, 128, 128],
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
    ):
        """Initilize global style encoder module."""
        super(StyleTokenEncoder, self).__init__()
        self.ref_enc = ReferenceEncoder(
            num_mels=num_mels,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=token_dim,
        )
        self.stl = StyleTokenLayer(
            num_tokens=num_tokens,
            token_dim=token_dim,
            num_heads=num_heads,
        )

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        """
        ref_embs = self.ref_enc(speech, speech_lengths)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module."""

    def __init__(
        self,
        num_mels=80,
        conv_layers: int = 6,
        conv_chans_list: List[int] = [32, 32, 64, 64, 128, 128],
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(**convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = num_mels
        for i in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, num_mels)
        hs = self.convs(xs)  # (B, conv_out_chans, Lmax', num_mels')
        # NOTE(kan-bayashi): We need to care the length?
        hlens = self._get_output_lengths(speech_lengths)
        hs = hs.transpose(1, 2).view(
            batch_size, hlens.max(), -1
        )  # (B, Lmax', gru_units)
        hs = pack_padded_sequence(hs, hlens, batch_first=True)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs

    def _get_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for i in range(self.conv_layers):
            lengths = (lengths - self.kernel_size + 2 * self.padding) // self.stride + 1
        return lengths


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module."""

    def __init__(
        self,
        num_tokens: int = 10,
        token_dim: int = 256,
        num_heads: int = 4,
    ):
        """Initilize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        self.register_parameter("gst_embs", torch.tensor(num_tokens, token_dim))
        self.mha = MultiHeadedAttention(num_heads, token_dim)

    def forward(
        self,
        ref_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Reference embeddings (B, token_dim).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = F.tanh(self.style_embs).unsqueeze(0).expand(batch_size, -1, -1)
        style_embs = self.mha(ref_embs.unsqueeze(1), gst_embs, gst_embs)

        return style_embs.squeeze(1)
