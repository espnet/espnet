# -*- coding: utf-8 -*-

# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration calculator for ESPnet2."""

from typing import Tuple

import torch


class DurationCalculator(torch.nn.Module):
    """
        Duration calculator for ESPnet2.

    This module implements a duration calculator that converts attention weights
    from a sequence-to-sequence model into durations and focus rate values.
    It can handle different input shapes corresponding to different model architectures,
    such as Tacotron 2 and Transformer-based models.

    Attributes:
        None

    Args:
        att_ws (torch.Tensor): Attention weight tensor with shape
            (T_feats, T_text) for Tacotron 2 or
            (#layers, #heads, T_feats, T_text) for Transformer models.

    Returns:
        Tuple[torch.LongTensor, torch.Tensor]: A tuple containing:
            - LongTensor: Duration of each input (T_text,).
            - Tensor: Focus rate value.

    Raises:
        ValueError: If `att_ws` is not a 2D or 4D tensor.

    Examples:
        >>> calculator = DurationCalculator()
        >>> att_ws_tacotron = torch.rand(100, 50)  # Example for Tacotron 2
        >>> duration, focus_rate = calculator(att_ws_tacotron)
        >>> print(duration.shape)  # Should print: torch.Size([50])
        >>> print(focus_rate)  # Focus rate value

        >>> att_ws_transformer = torch.rand(6, 8, 100, 50)  # Example for Transformer
        >>> duration, focus_rate = calculator(att_ws_transformer)
        >>> print(duration.shape)  # Should print: torch.Size([50])
        >>> print(focus_rate)  # Focus rate value
    """

    @torch.no_grad()
    def forward(self, att_ws: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Convert attention weight to durations.

        This method processes attention weights and computes the duration and focus rate
        from the given attention weight tensor. It supports both Tacotron 2 and transformer
        models based on the shape of the input tensor.

        Args:
            att_ws (torch.Tensor): Attention weight tensor. It can have one of the
                following shapes:
                - (T_feats, T_text) for Tacotron 2.
                - (#layers, #heads, T_feats, T_text) for transformer models.

        Returns:
            Tuple[torch.LongTensor, torch.Tensor]: A tuple containing:
                - Duration of each input (T_text,) as a LongTensor.
                - Focus rate value as a Tensor.

        Raises:
            ValueError: If `att_ws` is not a 2D or 4D tensor.

        Examples:
            >>> duration_calculator = DurationCalculator()
            >>> att_ws_tacotron = torch.rand(100, 50)  # Example for Tacotron 2
            >>> duration, focus_rate = duration_calculator(att_ws_tacotron)
            >>> print(duration.shape)  # Should print: torch.Size([50])
            >>> print(focus_rate)  # Focus rate value

            >>> att_ws_transformer = torch.rand(6, 8, 100, 50)  # Example for transformer
            >>> duration, focus_rate = duration_calculator(att_ws_transformer)
            >>> print(duration.shape)  # Should print: torch.Size([50])
            >>> print(focus_rate)  # Focus rate value
        """
        duration = self._calculate_duration(att_ws)
        focus_rate = self._calculate_focus_rete(att_ws)

        return duration, focus_rate

    @staticmethod
    def _calculate_focus_rete(att_ws):
        if len(att_ws.shape) == 2:
            # tacotron 2 case -> (T_feats, T_text)
            return att_ws.max(dim=-1)[0].mean()
        elif len(att_ws.shape) == 4:
            # transformer case -> (#layers, #heads, T_feats, T_text)
            return att_ws.max(dim=-1)[0].mean(dim=-1).max()
        else:
            raise ValueError("att_ws should be 2 or 4 dimensional tensor.")

    @staticmethod
    def _calculate_duration(att_ws):
        if len(att_ws.shape) == 2:
            # tacotron 2 case -> (T_feats, T_text)
            pass
        elif len(att_ws.shape) == 4:
            # transformer case -> (#layers, #heads, T_feats, T_text)
            # get the most diagonal head according to focus rate
            att_ws = torch.cat(
                [att_w for att_w in att_ws], dim=0
            )  # (#heads * #layers, T_feats, T_text)
            diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1)  # (#heads * #layers,)
            diagonal_head_idx = diagonal_scores.argmax()
            att_ws = att_ws[diagonal_head_idx]  # (T_feats, T_text)
        else:
            raise ValueError("att_ws should be 2 or 4 dimensional tensor.")
        # calculate duration from 2d attention weight
        durations = torch.stack(
            [att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])]
        )
        return durations.view(-1)
