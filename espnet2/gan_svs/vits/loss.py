# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SVS-related loss modules.

"""

import torch
import torch.nn.functional as F


class PitchLoss(torch.nn.Module):
    """Pitch loss."""

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate Pitch loss.

        Args:
            y_hat (Tensor): Generated pitch.
            y (Tensor): Groundtruth pitch tensor (B, 1, T).

        Returns:
            Tensor: Pitch loss.

        """
        pitch_loss = F.l1_loss(y_hat, y)
        return pitch_loss
