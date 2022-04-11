from abc import ABC
import logging


import ci_sdr
import fast_bss_eval
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


class SDRLoss(TimeDomainLoss):
    """SDR loss

    filter_length: int
        The length of the distortion filter allowed (default: ``512``)
    clamp_db: float
        clamp the output value in  [-clamp_db, clamp_db]
    zero_mean: bool
        When set to True, the mean of all signals is subtracted prior.

    """

    def __init__(self, filter_length=512, clamp_db=None, zero_mean=True):
        super().__init__()
        self.filter_length = filter_length
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean

    @property
    def name(self) -> str:
        return "sdr_loss"

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
        """args:

        ref: Tensor, (..., n_samples)
            reference signal
        est: Tensor (..., n_samples)
            estimated signal

        Returns
        -------
        loss: (...,)
            the SDR loss (negative sdr)
        """

        sdr_loss = fast_bss_eval.sdr_loss(
            est=est,
            ref=ref,
            filter_length=self.filter_length,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return sdr_loss


class SISNRLoss(TimeDomainLoss):
    """SI-SNR (SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.
    clamp_db: float
        clamp the output value in  [-clamp_db, clamp_db]
    zero_mean: bool
        When set to True, the mean of all signals is subtracted prior.
    eps: float
        Deprecated. Keeped for compatibility.
    """

    def __init__(self, clamp_db=None, zero_mean=True, eps=None):
        super().__init__()
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning("Eps is deprecated in si_snr loss, set clamp_db instead.")

    @property
    def name(self) -> str:
        return "si_snr_loss"

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
        """args:

        ref: Tensor, (..., n_samples)
            reference signal
        est: Tensor (..., n_samples)
            estimated signal

        Returns
        -------
        loss: (...,)
            the SI-SDR loss (negative si-sdr)
        """

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr


if __name__ == "__main__":

    ref = torch.rand(8, 16000)
    est = torch.rand(8, 16000)

    loss = SDRLoss(clamp_db=50, zero_mean=True)

    a = loss.forward(ref, est)

    print(a)
