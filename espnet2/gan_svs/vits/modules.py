#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch


class Projection(torch.nn.Module):
    """
        Projection is a PyTorch module that performs a linear transformation on the
    input tensor using a 1D convolutional layer. This module is typically used in
    the context of generative models, where it projects the hidden representation
    to a specified output dimension.

    Attributes:
        hidden_channels (int): The number of input channels for the projection.
        out_channels (int): The number of output channels for the projection.
        proj (torch.nn.Conv1d): A 1D convolutional layer that performs the
            projection.

    Args:
        hidden_channels (int): The number of input channels.
        out_channels (int): The number of output channels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - m_p (torch.Tensor): The mean parameters of the projected output.
            - logs_p (torch.Tensor): The log variance parameters of the projected
              output.

    Raises:
        ValueError: If the input tensor `x` and the mask `x_mask` are not of
        compatible shapes.

    Examples:
        >>> projection = Projection(hidden_channels=64, out_channels=32)
        >>> x = torch.randn(8, 64, 100)  # (B, attention_dim, T_text)
        >>> x_mask = torch.ones(8, 1, 100)  # Mask with shape (B, 1, T_text)
        >>> m_p, logs_p = projection(x, x_mask)
        >>> m_p.shape  # Should be (8, 32, 100)
        >>> logs_p.shape  # Should be (8, 32, 100)
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        """
            Forward pass for the Projection module.

        This method applies a 1D convolution to the input tensor `x` and
        computes the masked output statistics. The output consists of two
        tensors: `m_p` and `logs_p`, which represent the mean and log variance
        of the projected input, respectively.

        Args:
            x (torch.Tensor): Input tensor of shape (B, attention_dim, T_text).
            x_mask (torch.Tensor): Mask tensor of shape (B, 1, T_text) to
                apply masking during projection.

        Returns:
            tuple: A tuple containing:
                - m_p (torch.Tensor): The mean tensor of shape
                  (B, out_channels, T_text).
                - logs_p (torch.Tensor): The log variance tensor of shape
                  (B, out_channels, T_text).

        Examples:
            >>> projection = Projection(hidden_channels=128, out_channels=64)
            >>> x = torch.randn(32, 128, 50)  # Batch of 32, 128 features, 50 time steps
            >>> x_mask = torch.ones(32, 1, 50)  # Mask with all ones
            >>> m_p, logs_p = projection(x, x_mask)
            >>> m_p.shape
            torch.Size([32, 64, 50])
            >>> logs_p.shape
            torch.Size([32, 64, 50])

        Note:
            The input tensor `x` should be of appropriate shape and the mask
            should be broadcastable to match the dimensions of `x`.
        """
        # x shape: (B, attention_dim, T_text)
        stats = self.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
        return m_p, logs_p


def sequence_mask(length, max_length=None):
    """
        Creates a sequence mask for a given length tensor, which can be used to mask
    out padded values in sequences.

    This function generates a boolean mask tensor where the positions that are
    less than the lengths specified in the `length` tensor are marked as `True`,
    and positions that exceed these lengths are marked as `False`. This is useful
    in scenarios where you want to ignore padding in sequence processing.

    Args:
        length (torch.Tensor): A 1D tensor containing the lengths of sequences.
        max_length (int, optional): The maximum length for the mask. If not
            specified, it will be set to the maximum value in `length`.

    Returns:
        torch.Tensor: A 2D boolean tensor of shape (1, max_length) where each
            row corresponds to the mask for a sequence, indicating which positions
            are valid (True) and which are padding (False).

    Examples:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = sequence_mask(lengths)
        >>> print(mask)
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True,  True],
                [ True,  True, False, False, False]])

        >>> lengths = torch.tensor([4, 2, 3])
        >>> mask = sequence_mask(lengths, max_length=5)
        >>> print(mask)
        tensor([[ True,  True,  True,  True, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False]])
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
