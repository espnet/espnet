from abc import ABC
from typing import Tuple, Dict

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss


class TimeDomainLoss(AbsEnhLoss, ABC):
    pass



EPS = torch.finfo(torch.get_default_dtype()).eps


class SISNRLoss(TimeDomainLoss):

    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = float(eps)
        print(self.eps)
        
    @property
    def name(self) -> str:
        return 'si_snr'

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
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + self.eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + self.eps
        )
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.eps)  # [B]

        return -1 * pair_wise_si_snr
