from abc import ABC

import ci_sdr
import torch
from transformers import Wav2Vec2ForCTC, HubertForCTC
from asteroid.losses.stoi import NegSTOILoss
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss


class TimeDomainLoss(AbsEnhLoss, ABC):
    pass


EPS = torch.finfo(torch.get_default_dtype()).eps


class STOILoss(TimeDomainLoss):
    """STOI loss

    Reference:
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective
        Intelligibility Measure for Time-Frequency Weighted Noisy Speech',
        ICASSP 2010, Texas, Dallas.
        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for 
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', 
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.
        [3] J. Jensen and C. H. Taal, 'An Algorithm for Predicting the 
        Intelligibility of Speech Masked by Modulated Noise Maskers',
        IEEE Transactions on Audio, Speech and Language Processing, 2016.
        
    Args:
        ref: (Batch, samples)
        inf: (Batch, samples)
        filter_length (int): a time-invariant filter that allows
                                slight distortion via filtering
    Returns:
        loss: (Batch,)
    """

    def __init__(self, sr=16000):
        super().__init__()
        self.loss_stoi = NegSTOILoss(sr)

    @property
    def name(self) -> str:
        return "stoi_loss"

    def forward(self, ref: torch.Tensor, inf: torch.Tensor,) -> torch.Tensor:

        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        self.loss_stoi.to(inf.device)
        return self.loss_stoi(inf, ref)


class HubertDFLoss(TimeDomainLoss):
    """Hubert Deep Feature Loss

    Reference:
        Hsu, W. N., Bolte, B., Tsai, Y. H. H., Lakhotia, K., Salakhutdinov, R., 
        & Mohamed, A. (2021). HuBERT: Self-Supervised Speech Representation 
        Learning by Masked Prediction of Hidden Units. 
        arXiv preprint arXiv:2106.07447.
        pretrained model: https://huggingface.co/facebook/hubert-large-ls960-ft 
        
    Args:
        ref: (Batch, samples)
        inf: (Batch, samples)
        layer (int): calculate deep feature loss on this layer
    Returns:
        loss: (Batch,)
    """

    def __init__(self, layer=12):
        super().__init__()
        self.hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert.freeze_feature_extractor()
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.layer = layer

    @property
    def name(self) -> str:
        return "hubertdf_loss"

    def forward(self, ref: torch.Tensor, inf: torch.Tensor,) -> torch.Tensor:

        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        self.hubert.to(inf.device)
        act_target = self.hubert.hubert(ref, output_hidden_states=True)[-1][
            self.layer : self.layer + 1
        ]
        act_estimate = self.hubert.hubert(inf, output_hidden_states=True)[-1][
            self.layer : self.layer + 1
        ]
        return torch.log10(
            torch.stack(
                [
                    torch.sum((x - y) ** 2, dim=(1, 2))
                    for x, y in zip(act_target, act_estimate)
                ]
            )
            + 1
        )


class Wav2vec2DFLoss(TimeDomainLoss):
    """Wav2vec2 Deep Feature Loss

    Reference:
        Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0:
        A framework for self-supervised learning of speech representations.
        In Proc. NIPS 2020
        pretrained model: https://huggingface.co/facebook/wav2vec2-base-960h
        
    Args:
        ref: (Batch, samples)
        inf: (Batch, samples)
        layer (int): calculate deep feature loss on this layer
    Returns:
        loss: (Batch,)
    """

    def __init__(self, layer=12):
        super().__init__()
        self.w2v2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v2.freeze_feature_extractor()
        for param in self.w2v2.parameters():
            param.requires_grad = False
        self.layer = layer

    @property
    def name(self) -> str:
        return "w2v2df_loss"

    def forward(self, ref: torch.Tensor, inf: torch.Tensor,) -> torch.Tensor:

        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        self.w2v2.to(inf.device)
        act_target = self.w2v2.wav2vec2(ref, output_hidden_states=True)[-1][
            self.layer : self.layer + 1
        ]
        act_estimate = self.w2v2.wav2vec2(inf, output_hidden_states=True)[-1][
            self.layer : self.layer + 1
        ]
        return torch.log10(
            torch.stack(
                [
                    torch.sum((x - y) ** 2, dim=(1, 2))
                    for x, y in zip(act_target, act_estimate)
                ]
            )
            + 1
        )


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

    def forward(self, ref: torch.Tensor, inf: torch.Tensor,) -> torch.Tensor:

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

    def forward(self, ref: torch.Tensor, inf: torch.Tensor,) -> torch.Tensor:
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

    def forward(self, ref: torch.Tensor, inf: torch.Tensor,) -> torch.Tensor:
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
            torch.sum(s_target ** 2, dim=1, keepdim=True) + self.eps
        )  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + self.eps
        )
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.eps)  # [B]

        return -1 * pair_wise_si_snr
