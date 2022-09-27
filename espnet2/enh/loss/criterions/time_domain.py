import logging
import math
from abc import ABC

import ci_sdr
import fast_bss_eval
import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.layers.stft import Stft

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class TimeDomainLoss(AbsEnhLoss, ABC):
    """Base class for all time-domain Enhancement loss modules."""

    @property
    def name(self) -> str:
        return self._name

    @property
    def only_for_test(self) -> bool:
        return self._only_for_test

    @property
    def is_noise_loss(self) -> bool:
        return self._is_noise_loss

    @property
    def is_dereverb_loss(self) -> bool:
        return self._is_dereverb_loss

    def __init__(
        self,
        name,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        super().__init__()
        # only used during validation
        self._only_for_test = only_for_test
        # only used to calculate the noise-related loss
        self._is_noise_loss = is_noise_loss
        # only used to calculate the dereverberation-related loss
        self._is_dereverb_loss = is_dereverb_loss
        if is_noise_loss and is_dereverb_loss:
            raise ValueError(
                "`is_noise_loss` and `is_dereverb_loss` cannot be True at the same time"
            )
        if is_noise_loss and "noise" not in name:
            name = name + "_noise"
        if is_dereverb_loss and "dereverb" not in name:
            name = name + "_dereverb"
        self._name = name


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

    def __init__(
        self,
        filter_length=512,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "ci_sdr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.filter_length = filter_length

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
    def __init__(
        self,
        eps=EPS,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.eps = float(eps)

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        # the return tensor should be shape of (batch,)

        noise = inf - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )
        return -snr


class SDRLoss(TimeDomainLoss):
    """SDR loss.

    filter_length: int
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter:
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    clamp_db: float
        clamp the output value in  [-clamp_db, clamp_db]
    zero_mean: bool
        When set to True, the mean of all signals is subtracted prior.
    load_diag:
        If provided, this small value is added to the diagonal coefficients of
        the system metrices when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero
    """

    def __init__(
        self,
        filter_length=512,
        use_cg_iter=None,
        clamp_db=None,
        zero_mean=True,
        load_diag=None,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "sdr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.filter_length = filter_length
        self.use_cg_iter = use_cg_iter
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        self.load_diag = load_diag

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SDR forward.

        Args:
            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SDR loss (negative sdr)
        """

        sdr_loss = fast_bss_eval.sdr_loss(
            est=est,
            ref=ref,
            filter_length=self.filter_length,
            use_cg_iter=self.use_cg_iter,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            load_diag=self.load_diag,
            pairwise=False,
        )

        return sdr_loss


class SISNRLoss(TimeDomainLoss):
    """SI-SNR (or named SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.
    """

    def __init__(
        self,
        clamp_db=None,
        zero_mean=True,
        eps=None,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "si_snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning("Eps is deprecated in si_snr loss, set clamp_db instead.")
            if self.clamp_db is None:
                self.clamp_db = -math.log10(eps / (1 - eps)) * 10

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SI-SNR forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SI-SDR loss (negative si-sdr)
        """
        assert torch.is_tensor(est) and torch.is_tensor(ref), est

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr


class TimeDomainMSE(TimeDomainLoss):
    def __init__(
        self,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "TD_MSE_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

    def forward(self, ref, inf) -> torch.Tensor:
        """Time-domain MSE loss forward.

        Args:
            ref: (Batch, T) or (Batch, T, C)
            inf: (Batch, T) or (Batch, T, C)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        mseloss = (ref - inf).pow(2)
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 2:
            mseloss = mseloss.mean(dim=1)
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return mseloss


class TimeDomainL1(TimeDomainLoss):
    def __init__(
        self,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "TD_L1_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

    def forward(self, ref, inf) -> torch.Tensor:
        """Time-domain L1 loss forward.

        Args:
            ref: (Batch, T) or (Batch, T, C)
            inf: (Batch, T) or (Batch, T, C)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        l1loss = abs(ref - inf)
        if ref.dim() == 3:
            l1loss = l1loss.mean(dim=[1, 2])
        elif ref.dim() == 2:
            l1loss = l1loss.mean(dim=1)
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss


class MultiResL1SpecLoss(TimeDomainLoss):
    """Multi-Resolution L1 time-domain + STFT mag loss

    Reference:
    Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
    Towards Low-Distortion Multi-Channel Speech Enhancement:
    The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

    Attributes:
        window_sz: (list)
            list of STFT window sizes.
        hop_sz: (list, optional)
            list of hop_sizes, default is each window_sz // 2.
        eps: (float)
            stability epsilon
        time_domain_weight: (float)
            weight for time domain loss.
    """

    def __init__(
        self,
        window_sz=[512],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        name=None,
        only_for_test=False,
    ):
        _name = "TD_L1_loss" if name is None else name
        super(MultiResL1SpecLoss, self).__init__(_name, only_for_test=only_for_test)

        assert all([x % 2 == 0 for x in window_sz])
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.time_domain_weight = time_domain_weight
        self.eps = eps
        self.stft_encoders = torch.nn.ModuleList([])
        for w, h in zip(self.window_sz, self.hop_sz):
            stft_enc = Stft(
                n_fft=w,
                win_length=w,
                hop_length=h,
                window=None,
                center=True,
                normalized=False,
                onesided=True,
            )
            self.stft_encoders.append(stft_enc)

    @property
    def name(self) -> str:
        return "l1_timedomain+magspec_loss"

    def get_magnitude(self, stft):
        if is_torch_1_9_plus:
            stft = torch.complex(stft[..., 0], stft[..., 1])
        else:
            stft = ComplexTensor(stft[..., 0], stft[..., 1])

        return stft.abs()

    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ):
        """forward.

        Args:
            target: (Batch, T)
            estimate: (Batch, T)
        Returns:
            loss: (Batch,)
        """
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        # shape bsz, samples
        scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
            torch.sum(estimate**2, -1, keepdim=True) + self.eps
        )
        time_domain_loss = torch.sum((estimate * scaling_factor - target).abs(), dim=-1)

        if len(self.stft_encoders) == 0:
            return time_domain_loss
        else:
            spectral_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.stft_encoders:
                target_mag = self.get_magnitude(stft_enc(target)[0])
                estimate_mag = self.get_magnitude(
                    stft_enc(estimate * scaling_factor)[0]
                )
                c_loss = torch.sum((estimate_mag - target_mag).abs(), dim=(1, 2))
                spectral_loss += c_loss

            return time_domain_loss * self.time_domain_weight + (
                1 - self.time_domain_weight
            ) * spectral_loss / len(self.stft_encoders)
