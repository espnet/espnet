from abc import ABC
import logging

import ci_sdr
import fast_bss_eval
import torch


from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
import asteroid_filterbanks.transforms as af_transforms
from asteroid_filterbanks import make_enc_dec


class TimeDomainLoss(AbsEnhLoss, ABC):
    """Base class for all time-domain Enhancement loss modules."""

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

    def __init__(self, filter_length=512, name=None):
        super().__init__()
        self.filter_length = filter_length

        self._name = "ci_sdr_loss" if name is None else name

    @property
    def name(self) -> str:
        return self._name

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
    def __init__(self, eps=EPS, name=None):
        super().__init__()
        self.eps = float(eps)

        self._name = "snr_loss" if name is None else name

    @property
    def name(self) -> str:
        return self._name

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
        the system metrics when solving for the filter coefficients.
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
    ):
        super().__init__()

        self.filter_length = filter_length
        self.use_cg_iter = use_cg_iter
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        self.load_diag = load_diag

        self._name = "sdr_loss" if name is None else name

    @property
    def name(self) -> str:
        return self._name

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
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
            Deprecated. Keeped for compatibility.
    """

    def __init__(self, clamp_db=None, zero_mean=True, eps=None, name=None):
        super().__init__()
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning("Eps is deprecated in si_snr loss, set clamp_db instead.")

        self._name = "si_snr_loss" if name is None else name

    @property
    def name(self) -> str:
        return self._name

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
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

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr


class TimeDomainMSE(TimeDomainLoss):
    def __init__(self, name=None):
        super().__init__()
        self._name = "TD_MSE_loss" if name is None else name

    @property
    def name(self) -> str:
        return self._name

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
    def __init__(self, name=None):
        super().__init__()
        self._name = "TD_L1_loss" if name is None else name

    @property
    def name(self) -> str:
        return self._name

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
    def __init__(self, window_sz=[512], hop_sz=None, eps=1e-8, time_domain_weight=0.5):
        super(MultiResL1SpecLoss, self).__init__()

        assert [x % 2 == 0 for x in window_sz]
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.time_domain_weight = time_domain_weight
        self.eps = eps
        self.stft_encoders = torch.nn.ModuleList([])
        for w, h in zip(self.window_sz, self.hop_sz):
            stft_enc, _ = make_enc_dec("torch_stft", w, w, h)
            self.stft_encoders.append(stft_enc)

    @property
    def name(self) -> str:
        return "l1_timedomain+magspec_loss"

    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ):
        # shape bsz, samples
        scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
            torch.sum(estimate**2, -1, keepdim=True) + self.eps
        )
        time_domain_loss = torch.mean((estimate * scaling_factor - target).abs())

        if len(self.stft_encoders) == 0:
            return time_domain_loss
        else:
            spectral_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.stft_encoders:
                target_stft = stft_enc(target)
                estimate_stft = stft_enc(estimate * scaling_factor)
                c_loss = torch.mean(
                    (
                        af_transforms.mag(estimate_stft)
                        - af_transforms.mag(target_stft)
                    ).abs()
                )
                spectral_loss += c_loss

            return time_domain_loss * self.time_domain_weight + (
                1 - self.time_domain_weight
            ) * spectral_loss / len(self.stft_encoders)
