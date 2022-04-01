from abc import ABC

import ci_sdr
import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss


class TimeDomainLoss(AbsEnhLoss, ABC):
    pass


EPS = torch.finfo(torch.get_default_dtype()).eps


class CISDRLoss(TimeDomainLoss):
    """CI-SDR loss

    Reference:
        Convolutive Transfer Function Invariant SDR Training
        Criteria for Multi-Channel Reverberant Speech Separation;
        C. Boeddeker et al., 2021;
        https://arxiv.org/abs/2011.15003
    Args:
        ref: (Batch, samples)
        inf: (Batch, samples)
        filter_length (int): a time-invariant filter that allows
                                slight distortion via filtering
    Returns:
        loss: (Batch,)
    """

    def __init__(self, filter_length=512):
        super().__init__()
        self.filter_length = filter_length

    @property
    def name(self) -> str:
        return "ci_sdr_loss"

    def forward(
        self,
        ref: torch.Tensor,
        inf: torch.Tensor,
    ) -> torch.Tensor:

        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        return ci_sdr.pt.ci_sdr_loss(
            inf, ref, compute_permutation=False, filter_length=self.filter_length
        )


class SNRLoss(TimeDomainLoss):
    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "snr_loss"

    def forward(
        self,
        ref: torch.Tensor,
        inf: torch.Tensor,
    ) -> torch.Tensor:
        # the return tensor should be shape of (batch,)

        noise = inf - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )
        return -snr


class SISNRLoss(TimeDomainLoss):
    """
    A more stable SI-SNR loss with clamp from `fast_bss_eval`.
    Thanks to Robin Scheibler for the implementation.
    We should consider import `fast_bss_eval` in the future PR.
    """

    def __init__(self, clamp_db=None, zero_mean=True, eps=None):
        super().__init__()
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean

    @property
    def name(self) -> str:
        return "si_snr_loss"

    def _db_clamp_eps(self, db_max: float) -> float:
        """
        helper function to compute the clamping constant
        """
        e = 10.0 ** (-db_max / 10.0)
        eps = e / (1.0 + e)
        return eps

    def _coherence_to_neg_sdr(self, coh: torch.Tensor) -> torch.Tensor:
        """
        This function transforms the squared cosine to negative SDR value.
        If provided clamp_db will limit the output to the range [-clamp_db, clamp_db].
        """
        clamp_db = self.clamp_db

        if clamp_db is not None:
            # clamp within desired decibel range
            eps = self._db_clamp_eps(clamp_db)
        else:
            # theoretically the coh values should be in [0, 1],
            # so we clamp them there to avoid numerical issues.
            eps = 0.0
        coh = torch.clamp(coh, min=eps, max=(1 - eps))

        ratio = (1 - coh) / coh

        # apply the SDR mapping
        return 10.0 * torch.log10(ratio)

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
        """
        ref: Tensor, (..., n_samples)
            reference signal
        est: Tensor (..., n_samples)
            estimated signal
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]

        Returns
        -------
        loss: (...,)
            the SI-SDR loss (negative si-sdr)
        """

        assert ref.size() == est.size()

        if self.zero_mean:
            mean_ref = torch.mean(ref, dim=-1, keepdim=True)
            mean_est = torch.mean(est, dim=-1, keepdim=True)
            ref = ref - mean_ref
            est = est - mean_est

        ref = torch.nn.functional.normalize(ref, dim=-1)
        est = torch.nn.functional.normalize(est, dim=-1)
        cos_sq = torch.square(torch.einsum("...n,...n->...", ref, est))

        return self._coherence_to_neg_sdr(cos_sq)
