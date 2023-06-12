# -*- coding: utf-8 -*-

# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration calculator for ESPnet2."""

from typing import Tuple

import torch


class DurationCalculator(torch.nn.Module):
    """Duration calculator module."""

    @torch.no_grad()
    def forward(self, att_ws: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert attention weight to durations.

        Args:
            att_ws (Tesnor): Attention weight tensor (T_feats, T_text) or
                (#layers, #heads, T_feats, T_text).

        Returns:
            LongTensor: Duration of each input (T_text,).
            Tensor: Focus rate value.

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
