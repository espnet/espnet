# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS-related loss modules.

This code is based on https://github.com/jaywalnut310/vits.

"""

import torch
import torch.distributions as D


class KLDivergenceLoss(torch.nn.Module):
    """
    KL divergence loss.

    This module computes the Kullback-Leibler (KL) divergence loss between two
    distributions, which is commonly used in variational inference methods.

    The KL divergence measures how one probability distribution diverges from
    a second expected probability distribution.

    Attributes:
        None

    Args:
        z_p (Tensor): Flow hidden representation (B, H, T_feats).
        logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
        m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
        logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
        z_mask (Tensor): Mask tensor (B, 1, T_feats).

    Returns:
        Tensor: KL divergence loss.

    Examples:
        >>> kl_loss = KLDivergenceLoss()
        >>> z_p = torch.randn(32, 64, 100)
        >>> logs_q = torch.randn(32, 64, 100)
        >>> m_p = torch.randn(32, 64, 100)
        >>> logs_p = torch.randn(32, 64, 100)
        >>> z_mask = torch.ones(32, 1, 100)
        >>> loss = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        >>> print(loss)

    Note:
        This loss is useful for training generative models like VITS.

    Raises:
        None
    """

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


class KLDivergenceLossWithoutFlow(torch.nn.Module):
    """
        KL divergence loss without flow.

    This class implements the calculation of the Kullback-Leibler (KL) divergence
    loss in a variational inference framework, specifically designed for cases
    where flow-based representations are not used.

    Attributes:
        None

    Args:
        m_q (Tensor): Posterior encoder projected mean (B, H, T_feats).
        logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
        m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
        logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).

    Returns:
        Tensor: KL divergence loss, averaged over the batch.

    Examples:
        >>> kl_loss = KLDivergenceLossWithoutFlow()
        >>> m_q = torch.randn(32, 64, 100)  # Example tensor for m_q
        >>> logs_q = torch.randn(32, 64, 100)  # Example tensor for logs_q
        >>> m_p = torch.randn(32, 64, 100)  # Example tensor for m_p
        >>> logs_p = torch.randn(32, 64, 100)  # Example tensor for logs_p
        >>> loss = kl_loss(m_q, logs_q, m_p, logs_p)
        >>> print(loss)  # Output will be the calculated KL divergence loss
    """

    def forward(
        self,
        m_q: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss without flow.

        Args:
            m_q (Tensor): Posterior encoder projected mean (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
        """
        posterior_norm = D.Normal(m_q, torch.exp(logs_q))
        prior_norm = D.Normal(m_p, torch.exp(logs_p))
        loss = D.kl_divergence(posterior_norm, prior_norm).mean()
        return loss
