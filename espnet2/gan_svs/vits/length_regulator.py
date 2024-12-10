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
    """
        Length Regulator for adjusting mel-spectrogram lengths based on predicted durations.

    This module expands input mel-spectrograms to match the specified duration,
    allowing for control over the timing of the generated audio. The length
    regulator uses padding for batches with varying lengths and can incorporate
    state information during expansion.

    Attributes:
        pad_value (float): Value used for padding the output tensor.

    Args:
        pad_value (float, optional): Value used for padding. Default is 0.0.

    Methods:
        LR(x, duration, use_state_info=False):
            Length regulates input mel-spectrograms to match the specified duration.

        expand(batch, predicted, use_state_info=False):
            Expands input mel-spectrogram based on the predicted duration.

        forward(x, duration, use_state_info=False):
            Forward pass through the length regulator module.

    Examples:
        >>> length_regulator = LengthRegulator(pad_value=0.0)
        >>> mel_spectrogram = torch.randn(2, 80, 100)  # Example input
        >>> durations = torch.tensor([[5, 3], [4, 0]])  # Example durations
        >>> output, mel_lengths = length_regulator(mel_spectrogram, durations)

    Raises:
        ValueError: If the input tensors have incompatible dimensions.
    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.
        """
        super().__init__()
        self.pad_value = pad_value

    def LR(self, x, duration, use_state_info=False):
        """
                Length regulator related modules.

        This module implements a LengthRegulator class that regulates the length of
        input mel-spectrograms based on predicted durations. It allows for expansion
        of input sequences to match the desired lengths, with optional state
        information.

        Attributes:
            pad_value (float): Value used for padding in the output tensor.

        Args:
                    x (Tensor): Input tensor (B, dim, T).
                    duration (Tensor): Duration tensor (B, T).
                    use_state_info (bool, optional): Whether to use position information or not.

        Returns:
                    Tensor: Output tensor (B, dim, D_frame).
                    Tensor: Output length (B,).

        Examples:
            # Initialize the length regulator
            length_regulator = LengthRegulator(pad_value=0.0)

            # Define input tensor and duration tensor
            input_tensor = torch.rand(2, 80, 100)  # (B, dim, T)
            duration_tensor = torch.tensor([[1, 2, 1], [2, 1, 0]])  # (B, T)

            # Forward pass
            output, mel_len = length_regulator(input_tensor, duration_tensor)

            # Output shapes
            print(output.shape)  # Should be (B, dim, D_frame)
            print(mel_len.shape)  # Should be (B,)
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
        """
            Length Regulator for adjusting the duration of mel-spectrograms.

        This module expands input mel-spectrograms according to predicted
        durations, allowing for flexible length adjustment. It can also
        incorporate state information during the expansion process if desired.

        Attributes:
            pad_value (float): Value used for padding the output tensor.

            Args:
                batch (Tensor): Input tensor (T, dim).
                predicted (Tensor): Predicted duration tensor (T,).
                use_state_info (bool, optional): Whether to use position information or not.

            Returns:
                Tensor: Output tensor (D_frame, dim).

        Methods:
            LR(x, duration, use_state_info=False):
                Length regulates input mel-spectrograms to match the specified
                duration.

            expand(batch, predicted, use_state_info=False):
                Expands the input mel-spectrogram based on the predicted duration.

            forward(x, duration, use_state_info=False):
                Forward pass through the length regulator module.

        Examples:
            >>> length_regulator = LengthRegulator(pad_value=0.0)
            >>> x = torch.randn(2, 80, 100)  # Example input tensor
            >>> duration = torch.tensor([[1, 2, 1], [1, 1, 1]])  # Example durations
            >>> output, mel_len = length_regulator(x, duration)

        Note:
            This class is designed to work with PyTorch tensors and is
            intended for use in speech synthesis tasks.

        Raises:
            UserWarning: If the predicted durations include all zeros,
            the first element is filled with 1 to avoid zero-length sequences.
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
        """
        Forward pass through the length regulator module.

        This method takes an input tensor and a duration tensor, and processes
        them to generate an output tensor that has been length-regulated to
        match the specified durations.

        Args:
            x (Tensor): Input tensor of shape (B, dim, T), where B is the batch
                size, dim is the number of features, and T is the length of
                the input sequence.
            duration (Tensor): Duration tensor of shape (B, T), where each
                entry specifies the length to which the corresponding input
                sequence should be expanded.
            use_state_info (bool, optional): If set to True, the method will
                include positional information in the output. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Output tensor of shape (B, dim, D_frame), where D_frame is
                  the total length after expansion.
                - Output length tensor of shape (B,), which indicates the
                  length of the output for each batch.

        Raises:
            Warning: If the sum of durations is zero, a warning is logged and
            the first element of the duration tensor is set to 1 to avoid
            processing sequences with zero length.

        Examples:
            >>> length_regulator = LengthRegulator(pad_value=0.0)
            >>> x = torch.rand(2, 256, 50)  # Example input tensor
            >>> duration = torch.tensor([[1, 2, 3], [0, 4, 0]])  # Example durations
            >>> output, mel_len = length_regulator.forward(x, duration)
            >>> print(output.shape)  # Should reflect the regulated lengths
            >>> print(mel_len)  # Lengths of the output tensors for each batch
        """

        if duration.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            duration[duration.sum(dim=1).eq(0)] = 1

        output, mel_len = self.LR(x, duration, use_state_info=use_state_info)

        return output, mel_len
