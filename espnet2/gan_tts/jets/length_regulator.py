# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch


class GaussianUpsampling(torch.nn.Module):
    """
        Gaussian upsampling with fixed temperature as described in:

    https://arxiv.org/abs/2010.04301

    This module expands the hidden states based on the specified durations,
    using a Gaussian function to compute the attention weights for the
    upsampling process.

    Attributes:
        delta (float): The temperature parameter that controls the spread of the
            Gaussian function.

    Args:
        hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
        ds (Tensor): Batched token duration (B, T_text).
        h_masks (Tensor, optional): Mask tensor for hidden states (B, T_feats).
        d_masks (Tensor, optional): Mask tensor for durations (B, T_text).

    Returns:
        Tensor: Expanded hidden state (B, T_feat, adim).

    Raises:
        Warning: If the predicted durations include all zero sequences, a warning
            will be logged, and the first element will be filled with 1.

    Examples:
        >>> import torch
        >>> upsampler = GaussianUpsampling(delta=0.1)
        >>> hs = torch.randn(2, 5, 256)  # Example hidden states
        >>> ds = torch.tensor([[1, 2, 0], [2, 1, 1]])  # Example durations
        >>> expanded_hs = upsampler(hs, ds)
        >>> print(expanded_hs.shape)
        torch.Size([2, T_feat, 256])  # Shape will depend on the durations

    Note:
        The behavior of this function may differ based on the input masks and
        the provided durations.
    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds, h_masks=None, d_masks=None):
        """
            Gaussian upsampling with fixed temperature as in:

        https://arxiv.org/abs/2010.04301

        Attributes:
            delta (float): Temperature parameter for the Gaussian function.

        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).
            h_masks (Tensor, optional): Mask tensor (B, T_feats). Default is None.
            d_masks (Tensor, optional): Mask tensor (B, T_text). Default is None.

        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).

        Raises:
            Warning: If the predicted durations include all zero sequences.

        Examples:
            >>> model = GaussianUpsampling(delta=0.1)
            >>> hs = torch.randn(2, 5, 10)  # Example hidden states
            >>> ds = torch.tensor([[1, 2, 0, 0, 0], [1, 0, 0, 0, 0]])  # Durations
            >>> output = model.forward(hs, ds)
            >>> print(output.shape)  # Should print the shape of expanded hidden states

        Note:
            The method handles cases where the duration tensor contains all zeros
            by filling the first element with 1, as this situation should not occur
            during teacher forcing.
        """
        B = ds.size(0)
        device = ds.device

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        if h_masks is None:
            T_feats = ds.sum().int()
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()
        if h_masks is not None:
            t = t * h_masks.float()

        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(
                ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
            )

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        hs = torch.matmul(p_attn, hs)
        return hs
