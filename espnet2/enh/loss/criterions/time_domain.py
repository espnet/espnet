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
    """
    Base class for all time-domain Enhancement loss modules.

    This class serves as a base for various time-domain loss functions used
    in speech enhancement tasks. It includes properties to specify the type
    of loss, whether it's for testing, noise-related, or dereverberation-related.

    Attributes:
        name (str): The name of the loss function.
        only_for_test (bool): Flag indicating if the loss is only for testing.
        is_noise_loss (bool): Flag indicating if the loss is related to noise.
        is_dereverb_loss (bool): Flag indicating if the loss is related to
            dereverberation.

    Args:
        name (str): Name of the loss function.
        only_for_test (bool): Optional; defaults to False.
        is_noise_loss (bool): Optional; defaults to False.
        is_dereverb_loss (bool): Optional; defaults to False.

    Raises:
        ValueError: If both `is_noise_loss` and `is_dereverb_loss` are True
            at the same time, or if the name does not contain the appropriate
            suffix when `is_noise_loss` or `is_dereverb_loss` is set to True.

    Examples:
        >>> loss = TimeDomainLoss(name="my_loss", is_noise_loss=True)
        >>> print(loss.name)  # Output: my_loss_noise
    """

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
    """
    CI-SDR loss.

    This class implements the Convolutive Transfer Function Invariant SDR
    Training Criteria for Multi-Channel Reverberant Speech Separation. It
    computes the CI-SDR loss between reference and estimated signals, allowing
    for a slight distortion via a time-invariant filter.

    Reference:
        Convolutive Transfer Function Invariant SDR Training
        Criteria for Multi-Channel Reverberant Speech Separation;
        C. Boeddeker et al., 2021;
        https://arxiv.org/abs/2011.15003

    Args:
        filter_length (int): A time-invariant filter length that allows
            slight distortion via filtering (default: 512).
        name (str, optional): Name of the loss function. If None, defaults to
            "ci_sdr_loss".
        only_for_test (bool, optional): Indicates if the loss is only for
            testing (default: False).
        is_noise_loss (bool, optional): Indicates if this is a noise-related
            loss (default: False).
        is_dereverb_loss (bool, optional): Indicates if this is a
            dereverberation-related loss (default: False).

    Returns:
        torch.Tensor: The computed loss of shape (Batch,).

    Raises:
        ValueError: If both `is_noise_loss` and `is_dereverb_loss` are
        True.

    Examples:
        >>> import torch
        >>> loss_fn = CISDRLoss(filter_length=256)
        >>> reference = torch.randn(10, 16000)  # 10 samples, 16000 time steps
        >>> estimated = torch.randn(10, 16000)
        >>> loss = loss_fn(reference, estimated)
        >>> print(loss.shape)  # Output: torch.Size([10])
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
        """
            Compute the CI-SDR loss between reference and estimated signals.

        This method calculates the CI-SDR (Convolutive Transfer Function Invariant
        Signal-to-Distortion Ratio) loss, which is a metric used for evaluating the
        quality of audio signals after separation or enhancement.

        Args:
            ref (torch.Tensor): The reference signal tensor with shape (Batch, samples).
            inf (torch.Tensor): The estimated signal tensor with shape (Batch, samples).

        Returns:
            torch.Tensor: The computed CI-SDR loss tensor with shape (Batch,).

        Raises:
            AssertionError: If the shapes of `ref` and `inf` do not match.

        Examples:
            >>> import torch
            >>> loss_fn = CISDRLoss()
            >>> ref = torch.randn(8, 16000)  # Batch of 8 signals, each 16000 samples
            >>> inf = torch.randn(8, 16000)  # Estimated signals
            >>> loss = loss_fn.forward(ref, inf)
            >>> print(loss.shape)  # Output: torch.Size([8])
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        return ci_sdr.pt.ci_sdr_loss(
            inf, ref, compute_permutation=False, filter_length=self.filter_length
        )


class SNRLoss(TimeDomainLoss):
    """
    SNR (Signal-to-Noise Ratio) loss for time-domain enhancement.

    This loss computes the negative signal-to-noise ratio (SNR) between the
    reference signal and the estimated signal. The SNR is calculated in
    decibels, where a higher SNR indicates better performance.

    Args:
        eps (float): A small constant added to the denominator to avoid
                     division by zero (default: machine epsilon).
        name (str, optional): Name of the loss function (default: "snr_loss").
        only_for_test (bool, optional): If True, this loss is only used during
                                         testing (default: False).
        is_noise_loss (bool, optional): If True, this loss is used for noise
                                          related tasks (default: False).
        is_dereverb_loss (bool, optional): If True, this loss is used for
                                            dereverberation tasks (default: False).

    Returns:
        torch.Tensor: The computed SNR loss, with shape (Batch,).

    Examples:
        >>> snr_loss = SNRLoss()
        >>> reference = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        >>> estimate = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        >>> loss = snr_loss(reference, estimate)
        >>> print(loss)
        tensor([-3.0103, -3.0103])  # Example output, will vary based on input.

    Raises:
        AssertionError: If the shapes of the reference and estimated tensors do not match.
    """

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
        """
            Signal-to-Noise Ratio (SNR) Loss.

        This class computes the Signal-to-Noise Ratio (SNR) loss, which is defined as
        the difference in decibels between the signal and the noise. The SNR is a
        critical metric in evaluating the quality of audio signals, especially in
        speech enhancement tasks.

        Attributes:
            eps (float): A small value to prevent division by zero.

        Args:
            eps (float): A small value to avoid numerical instability (default:
                         machine epsilon).
            name (str, optional): Name of the loss function (default: "snr_loss").
            only_for_test (bool, optional): Flag indicating if the loss is only for
                                             testing (default: False).
            is_noise_loss (bool, optional): Flag indicating if this loss is for noise
                                              estimation (default: False).
            is_dereverb_loss (bool, optional): Flag indicating if this loss is for
                                                dereverberation (default: False).

        Raises:
            ValueError: If both `is_noise_loss` and `is_dereverb_loss` are True.

        Examples:
            >>> import torch
            >>> snr_loss = SNRLoss()
            >>> reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> estimated = torch.tensor([[1.0, 2.5], [2.5, 4.5]])
            >>> loss = snr_loss(reference, estimated)
            >>> print(loss)
            tensor([-2.0000, -2.0000])

        Notes:
            The returned loss value is negative, representing the loss to be
            minimized during training.
        """
        # the return tensor should be shape of (batch,)

        noise = inf - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )
        return -snr


class SDRLoss(TimeDomainLoss):
    """
    SDR loss.

    This class computes the Signal-to-Distortion Ratio (SDR) loss, which is
    commonly used in speech enhancement tasks. The SDR loss is useful for
    measuring the quality of an estimated signal compared to a reference
    signal, focusing on minimizing the distortion introduced by the
    enhancement process.

    Attributes:
        filter_length (int): The length of the distortion filter allowed
            (default: ``512``).
        use_cg_iter (int or None): If provided, an iterative method is
            used to solve for the distortion filter coefficients instead
            of direct Gaussian elimination. This can speed up the
            computation of the metrics in case the filters are long.
            Using a value of 10 here has been shown to provide good
            accuracy in most cases and is sufficient when using this
            loss to train neural separation networks.
        clamp_db (float or None): Clamp the output value in
            [-clamp_db, clamp_db].
        zero_mean (bool): When set to True, the mean of all signals
            is subtracted prior to calculation.
        load_diag (float or None): If provided, this small value is
            added to the diagonal coefficients of the system matrices
            when solving for the filter coefficients. This can help
            stabilize the metric in the case where some of the reference
            signals may sometimes be zero.

    Args:
        filter_length (int, optional): Length of the distortion filter
            (default: 512).
        use_cg_iter (int or None, optional): Iterative method for
            solving filter coefficients (default: None).
        clamp_db (float or None, optional): Clamping value for output
            (default: None).
        zero_mean (bool, optional): Whether to zero the mean (default: True).
        load_diag (float or None, optional): Small value for diagonal
            stabilization (default: None).
        name (str, optional): Name of the loss function (default: None).
        only_for_test (bool, optional): If the loss is only for testing
            (default: False).
        is_noise_loss (bool, optional): If the loss is noise-related
            (default: False).
        is_dereverb_loss (bool, optional): If the loss is related to
            dereverberation (default: False).

    Returns:
        torch.Tensor: The computed SDR loss (negative SDR).

    Examples:
        >>> import torch
        >>> sdr_loss = SDRLoss()
        >>> reference = torch.randn(2, 512)  # Batch of 2 signals
        >>> estimated = torch.randn(2, 512)  # Estimated signals
        >>> loss = sdr_loss(reference, estimated)
        >>> print(loss)  # Output will be the SDR loss value

    Note:
        Ensure that the reference and estimated tensors have the same shape
        when passing them to the forward method.
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
        """
        Calculate the SDR loss.

        This method computes the SDR (Signal-to-Distortion Ratio) loss
        between the reference signal and the estimated signal. The
        SDR loss is calculated as the negative SDR value.

        Args:
            ref: Tensor of shape (..., n_samples)
                The reference signal.
            est: Tensor of shape (..., n_samples)
                The estimated signal.

        Returns:
            loss: Tensor of shape (...)
                The SDR loss (negative SDR).

        Examples:
            >>> import torch
            >>> sdr_loss = SDRLoss()
            >>> ref = torch.randn(2, 1000)  # Example reference signals
            >>> est = torch.randn(2, 1000)  # Example estimated signals
            >>> loss = sdr_loss(ref, est)
            >>> print(loss.shape)  # Output: torch.Size([2])
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
    """
    SI-SNR (or SI-SDR) loss.

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            Clamp the output value in [-clamp_db, clamp_db].
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.

    Args:
        clamp_db: (float, optional)
            Clamp the output value in [-clamp_db, clamp_db]. Default is None.
        zero_mean: (bool, optional)
            When set to True, the mean of all signals is subtracted prior.
            Default is True.
        eps: (float, optional)
            Deprecated parameter for compatibility. Default is None.
        name: (str, optional)
            Name of the loss function. Default is "si_snr_loss".
        only_for_test: (bool, optional)
            If True, the loss is only used during testing. Default is False.
        is_noise_loss: (bool, optional)
            If True, the loss is related to noise. Default is False.
        is_dereverb_loss: (bool, optional)
            If True, the loss is related to dereverberation. Default is False.

    Returns:
        loss: (torch.Tensor)
            The SI-SDR loss (negative SI-SDR).

    Examples:
        >>> loss_fn = SISNRLoss(clamp_db=10)
        >>> reference = torch.randn(8, 16000)  # (Batch, samples)
        >>> estimated = torch.randn(8, 16000)  # (Batch, samples)
        >>> loss = loss_fn(reference, estimated)
        >>> print(loss.shape)  # Should output: torch.Size([8])

    Note:
        The `eps` parameter is deprecated and will be removed in future versions.
        It is recommended to use `clamp_db` instead.
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
        """
            SI-SNR (or named SI-SDR) loss.

        A more stable SI-SNR loss with clamp from `fast_bss_eval`.

        Attributes:
            clamp_db: float
                Clamp the output value in [-clamp_db, clamp_db].
            zero_mean: bool
                When set to True, the mean of all signals is subtracted prior.
            eps: float
                Deprecated. Kept for compatibility.

        Args:
            clamp_db: (float, optional)
                Clamp the output value in [-clamp_db, clamp_db]. Default is None.
            zero_mean: (bool, optional)
                When set to True, the mean of all signals is subtracted prior.
                Default is True.
            eps: (float, optional)
                Deprecated. Kept for compatibility. Default is None.
            name: (str, optional)
                Name of the loss function. Default is "si_snr_loss".
            only_for_test: (bool, optional)
                Flag to indicate if the loss is only for testing. Default is False.
            is_noise_loss: (bool, optional)
                Flag to indicate if the loss is related to noise. Default is False.
            is_dereverb_loss: (bool, optional)
                Flag to indicate if the loss is related to dereverberation. Default is
                False.

        Returns:
            loss: (torch.Tensor)
                The SI-SDR loss (negative SI-SDR).

        Examples:
            >>> loss_function = SISNRLoss(clamp_db=10.0)
            >>> reference_signal = torch.randn(1, 16000)
            >>> estimated_signal = torch.randn(1, 16000)
            >>> loss = loss_function(reference_signal, estimated_signal)
            >>> print(loss)

        Note:
            The parameter `eps` is deprecated; it is recommended to use `clamp_db`
            instead for stability.
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
    """
    Time-domain Mean Squared Error (MSE) loss.

    This loss computes the mean squared error between the reference signal
    and the estimated signal in the time domain. It can handle inputs
    with different dimensions, either 2D (Batch, T) or 3D (Batch, T, C).

    Args:
        name (str, optional): The name of the loss function. Default is
            "TD_MSE_loss".
        only_for_test (bool, optional): Flag to indicate if the loss is only
            used for testing. Default is False.
        is_noise_loss (bool, optional): Flag to indicate if this loss is
            related to noise. Default is False.
        is_dereverb_loss (bool, optional): Flag to indicate if this loss is
            related to dereverberation. Default is False.

    Raises:
        ValueError: If the shapes of the reference and estimated signals do
            not match.

    Returns:
        torch.Tensor: The computed loss, shape (Batch,).

    Examples:
        >>> import torch
        >>> loss_fn = TimeDomainMSE()
        >>> ref = torch.randn(4, 16000)  # 4 samples, 16000 time steps
        >>> inf = torch.randn(4, 16000)
        >>> loss = loss_fn(ref, inf)
        >>> print(loss.shape)  # Output: torch.Size([4])
    """

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
        """
            Time-domain MSE loss forward.

        Computes the Mean Squared Error (MSE) loss between the reference signal
        and the estimated signal in the time domain. The MSE is calculated as
        the average of the squared differences between the two signals.

        Args:
            ref: (Batch, T) or (Batch, T, C)
                The reference signal(s) for comparison.
            inf: (Batch, T) or (Batch, T, C)
                The estimated signal(s) to be evaluated against the reference.

        Returns:
            loss: (Batch,)
                The computed MSE loss for each item in the batch.

        Raises:
            ValueError: If the shapes of `ref` and `inf` do not match.

        Examples:
            >>> import torch
            >>> ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> inf = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
            >>> loss = TimeDomainMSE().forward(ref, inf)
            >>> print(loss)  # Output: tensor([0.25, 0.25])
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
    """
    Time-domain L1 loss.

    This loss function computes the L1 loss (Mean Absolute Error) between
    the reference and estimated signals in the time domain. It is often used
    in tasks such as speech enhancement where preserving the structure of
    the waveform is essential.

    Args:
        name (str, optional): The name of the loss function. Defaults to
            "TD_L1_loss".
        only_for_test (bool, optional): If True, the loss is only used for
            testing. Defaults to False.
        is_noise_loss (bool, optional): If True, indicates that this loss
            is specifically for noise-related tasks. Defaults to False.
        is_dereverb_loss (bool, optional): If True, indicates that this loss
            is specifically for dereverberation tasks. Defaults to False.

    Returns:
        torch.Tensor: The computed L1 loss with shape (Batch,).

    Raises:
        ValueError: If the shapes of the reference and estimated signals do
            not match.

    Examples:
        >>> import torch
        >>> loss_fn = TimeDomainL1()
        >>> ref = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> inf = torch.tensor([[1.0, 2.5, 3.0], [3.5, 5.0, 6.0]])
        >>> loss = loss_fn(ref, inf)
        >>> print(loss)
        tensor([0.1667, 0.1667])
    """

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
        """
            Time-domain L1 loss forward.

        This method computes the L1 loss between the reference and estimated
        signals in the time domain. The L1 loss is defined as the mean absolute
        difference between the two signals.

        Args:
            ref: (Batch, T) or (Batch, T, C)
                The reference signal tensor, which can be either 2D or 3D.
            inf: (Batch, T) or (Batch, T, C)
                The estimated signal tensor, which should match the shape of `ref`.

        Returns:
            loss: (Batch,)
                The computed L1 loss for each sample in the batch.

        Raises:
            ValueError: If the shapes of `ref` and `inf` do not match.

        Examples:
            >>> import torch
            >>> loss_fn = TimeDomainL1()
            >>> ref = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> inf = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
            >>> loss = loss_fn(ref, inf)
            >>> print(loss)
            tensor([0.5000, 0.5000])

        Note:
            The method ensures that the input tensors are of compatible shapes
            before proceeding with the loss computation.
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
    """
    Multi-Resolution L1 time-domain + STFT magnitude loss.

    This loss function combines the L1 loss in the time domain with the
    short-time Fourier transform (STFT) magnitude loss. It aims to improve
    the quality of speech enhancement by leveraging both time-domain and
    frequency-domain information.

    Reference:
        Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... &
        Watanabe, S. Towards Low-Distortion Multi-Channel Speech Enhancement:
        The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022
        p. 9201-9205.

    Attributes:
        window_sz (list): A list of STFT window sizes.
        hop_sz (list, optional): A list of hop sizes, default is each
            window_sz // 2.
        eps (float): Stability epsilon to prevent division by zero.
        time_domain_weight (float): Weight for time domain loss.
        normalize_variance (bool): Whether to normalize the variance when
            calculating the loss.
        reduction (str): Method for reducing the loss, select from "sum"
            and "mean".

    Args:
        window_sz (list): List of STFT window sizes.
        hop_sz (list, optional): List of hop sizes, default is
            each window_sz // 2.
        eps (float, optional): Stability epsilon (default: 1e-8).
        time_domain_weight (float, optional): Weight for time domain loss
            (default: 0.5).
        normalize_variance (bool, optional): Normalize variance when
            calculating loss (default: False).
        reduction (str, optional): Reduction method, either "sum" or
            "mean" (default: "sum").
        name (str, optional): Name of the loss function (default: None).
        only_for_test (bool, optional): If True, only used for testing
            (default: False).
        is_noise_loss (bool, optional): If True, the loss is related to
            noise (default: False).
        is_dereverb_loss (bool, optional): If True, the loss is related to
            dereverberation (default: False).

    Returns:
        torch.Tensor: The computed loss value with shape (Batch,).

    Examples:
        >>> loss_fn = MultiResL1SpecLoss(window_sz=[256, 512])
        >>> target = torch.randn(10, 16000)  # (Batch, T)
        >>> estimate = torch.randn(10, 16000)  # (Batch, T)
        >>> loss = loss_fn(target, estimate)
        >>> print(loss.shape)  # Output: torch.Size([10])
    """

    def __init__(
        self,
        window_sz=[512],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
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

        assert all([x % 2 == 0 for x in window_sz])
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.time_domain_weight = time_domain_weight
        self.normalize_variance = normalize_variance
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

        assert reduction in ("sum", "mean")
        self.reduction = reduction

    @property
    def name(self) -> str:
        return "l1_timedomain+magspec_loss"

    def get_magnitude(self, stft, eps=1e-06):
        """
            Multi-Resolution L1 time-domain + STFT magnitude loss.

        This loss function combines the time-domain L1 loss with the
        Short-Time Fourier Transform (STFT) magnitude loss to enhance
        speech signals by minimizing distortion. It is particularly useful
        for multi-channel speech enhancement tasks.

        Reference:
            Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z.,
            ... & Watanabe, S. Towards Low-Distortion Multi-Channel Speech
            Enhancement: The ESPNET-Se Submission to the L3DAS22 Challenge.
            ICASSP 2022 p. 9201-9205.

        Attributes:
            window_sz: (list)
                List of STFT window sizes.
            hop_sz: (list, optional)
                List of hop sizes, default is each window_sz // 2.
            eps: (float)
                Stability epsilon.
            time_domain_weight: (float)
                Weight for time-domain loss.
            normalize_variance: (bool)
                Whether or not to normalize the variance when calculating the loss.
            reduction: (str)
                Select from "sum" and "mean".

        Args:
            window_sz: List of integers representing window sizes for STFT.
            hop_sz: Optional list of integers representing hop sizes.
            eps: Float for numerical stability.
            time_domain_weight: Float for weighting time-domain loss.
            normalize_variance: Boolean to normalize variance.
            reduction: String to specify reduction method, either "sum" or "mean".
            name: Optional name for the loss instance.
            only_for_test: Boolean to indicate if the loss is for testing only.
            is_noise_loss: Boolean to indicate if the loss is for noise-related tasks.
            is_dereverb_loss: Boolean to indicate if the loss is for dereverberation tasks.

        Methods:
            get_magnitude(stft, eps=1e-06):
                Computes the magnitude of the STFT.

        Examples:
            >>> loss = MultiResL1SpecLoss(window_sz=[256, 512], time_domain_weight=0.7)
            >>> target = torch.randn(10, 512)
            >>> estimate = torch.randn(10, 512)
            >>> output = loss(target, estimate)

        Raises:
            AssertionError: If the input dimensions do not match in the forward method.
        """
        if is_torch_1_9_plus:
            stft = torch.complex(stft[..., 0], stft[..., 1])
            return stft.abs()
        else:
            stft = ComplexTensor(stft[..., 0], stft[..., 1])
            return (stft.real.pow(2) + stft.imag.pow(2) + eps).sqrt()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ):
        """
            Multi-Resolution L1 time-domain + STFT magnitude loss.

        This loss combines time-domain L1 loss with the STFT magnitude loss
        to achieve low-distortion multi-channel speech enhancement. It
        effectively leverages both time-domain and frequency-domain
        information to improve the quality of speech signals.

        Reference:
            Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... &
            Watanabe, S. Towards Low-Distortion Multi-Channel Speech Enhancement:
            The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022
            p. 9201-9205.

        Attributes:
            window_sz (list): A list of STFT window sizes.
            hop_sz (list, optional): A list of hop sizes; defaults to
                each window_sz // 2.
            eps (float): Stability epsilon to avoid division by zero.
            time_domain_weight (float): Weight for time-domain loss.
            normalize_variance (bool): Whether to normalize the variance
                when calculating the loss.
            reduction (str): Select from "sum" and "mean" for loss reduction.

        Args:
            window_sz (list): List of STFT window sizes.
            hop_sz (list, optional): List of hop sizes; defaults to
                each window_sz // 2.
            eps (float, optional): Stability epsilon; defaults to 1e-8.
            time_domain_weight (float, optional): Weight for time-domain
                loss; defaults to 0.5.
            normalize_variance (bool, optional): Whether to normalize
                variance; defaults to False.
            reduction (str, optional): Method of reduction; defaults to
                "sum".
            name (str, optional): Name of the loss; defaults to None.
            only_for_test (bool, optional): Flag for test-only mode;
                defaults to False.
            is_noise_loss (bool, optional): Flag indicating if it's a
                noise loss; defaults to False.
            is_dereverb_loss (bool, optional): Flag indicating if it's a
                dereverberation loss; defaults to False.

        Returns:
            torch.Tensor: The computed loss with shape (Batch,).

        Examples:
            >>> loss_fn = MultiResL1SpecLoss(window_sz=[256, 512],
            ...                                hop_sz=[128, 256])
            >>> target = torch.randn(8, 512)
            >>> estimate = torch.randn(8, 512)
            >>> loss = loss_fn(target, estimate)
            >>> print(loss.shape)
            torch.Size([8])
        """
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        half_precision = (torch.float16, torch.bfloat16)
        if target.dtype in half_precision or estimate.dtype in half_precision:
            target = target.float()
            estimate = estimate.float()
        if self.normalize_variance:
            target = target / torch.std(target, dim=1, keepdim=True)
            estimate = estimate / torch.std(estimate, dim=1, keepdim=True)
        # shape bsz, samples
        scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
            torch.sum(estimate**2, -1, keepdim=True) + self.eps
        )
        if self.reduction == "sum":
            time_domain_loss = torch.sum(
                (estimate * scaling_factor - target).abs(), dim=-1
            )
        elif self.reduction == "mean":
            time_domain_loss = torch.mean(
                (estimate * scaling_factor - target).abs(), dim=-1
            )

        if len(self.stft_encoders) == 0:
            return time_domain_loss
        else:
            spectral_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.stft_encoders:
                target_mag = self.get_magnitude(stft_enc(target)[0])
                estimate_mag = self.get_magnitude(
                    stft_enc(estimate * scaling_factor)[0]
                )
                if self.reduction == "sum":
                    c_loss = torch.sum((estimate_mag - target_mag).abs(), dim=(1, 2))
                elif self.reduction == "mean":
                    c_loss = torch.mean((estimate_mag - target_mag).abs(), dim=(1, 2))
                spectral_loss += c_loss

            return time_domain_loss * self.time_domain_weight + (
                1 - self.time_domain_weight
            ) * spectral_loss / len(self.stft_encoders)
