# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS-related loss modules.

This code is based on https://github.com/jaywalnut310/vits.

"""

import torch


class KLDivergenceLoss(torch.nn.Module):
    """KL divergence loss."""

    def forward(
        self,
        z_p: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss.

        Args:
            z_p (Tensor): Flow hidden representation (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
            z_mask (Tensor): Mask tensor (B, 1, T_feats).

        Returns:
            Tensor: KL divergence loss.

        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        loss = kl / torch.sum(z_mask)

        return loss
