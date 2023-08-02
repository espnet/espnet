#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import logging

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


class LengthRegulator(torch.nn.Module):
    """Length Regulator"""

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.
        """
        super().__init__()
        self.pad_value = pad_value

    def LR(self, x, duration, use_state_info=False):
        """Length regulates input mel-spectrograms to match duration.

        Args:
            x (Tensor): Input tensor (B, dim, T).
            duration (Tensor): Duration tensor (B, T).
            use_state_info (bool, optional): Whether to use position information or not.

        Returns:
            Tensor: Output tensor (B, dim, D_frame).
            Tensor: Output length (B,).
        """
        x = torch.transpose(x, 1, 2)
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target, use_state_info=use_state_info)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        output = pad_list(output, self.pad_value)  # (B, D_frame, dim)
        output = torch.transpose(output, 1, 2)
        return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted, use_state_info=False):
        """Expand input mel-spectrogram based on the predicted duration.

        Args:
            batch (Tensor): Input tensor (T, dim).
            predicted (Tensor): Predicted duration tensor (T,).
            use_state_info (bool, optional): Whether to use position information or not.

        Returns:
            Tensor: Output tensor (D_frame, dim).
        """
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            if use_state_info:
                state_info_index = torch.unsqueeze(
                    torch.arange(0, expand_size), 1
                ).float()
                state_info_length = torch.unsqueeze(
                    torch.Tensor([expand_size] * expand_size), 1
                ).float()
                state_info = torch.cat([state_info_index, state_info_length], 1).to(
                    vec.device
                )
            new_vec = vec.expand(max(int(expand_size), 0), -1)
            if use_state_info:
                new_vec = torch.cat([new_vec, state_info], 1)

            out.append(new_vec)
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, use_state_info=False):
        """Forward pass through the length regulator module.

        Args:
            x (Tensor): Input tensor (B, dim, T).
            duration (Tensor): Duration tensor (B, T).
            use_state_info (bool, optional): Whether to use position information or not.

        Returns:
            Tensor: Output tensor (B, dim, D_frame).
            Tensor: Output length (B,).
        """

        if duration.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            duration[duration.sum(dim=1).eq(0)] = 1

        output, mel_len = self.LR(x, duration, use_state_info=use_state_info)

        return output, mel_len
