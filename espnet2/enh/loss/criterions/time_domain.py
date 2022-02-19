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
    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "si_snr_loss"

    def forward(
        self,
        ref: torch.Tensor,
        inf: torch.Tensor,
    ) -> torch.Tensor:
        # the return tensor should be shape of (batch,)
        assert ref.size() == inf.size()
        B, T = ref.size()

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = (
            torch.sum(s_target**2, dim=1, keepdim=True) + self.eps
        )  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj**2, dim=1) / (
            torch.sum(e_noise**2, dim=1) + self.eps
        )
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.eps)  # [B]

        return -1 * pair_wise_si_snr
