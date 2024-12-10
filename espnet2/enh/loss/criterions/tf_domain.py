import math
from abc import ABC, abstractmethod
from functools import reduce

import torch
import torch.nn.functional as F
from packaging.version import parse as V

from espnet2.enh.layers.complex_utils import complex_norm, is_complex, new_complex_like
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


def _create_mask_label(mix_spec, ref_spec, noise_spec=None, mask_type="IAM"):
    """Create mask label.

    Args:
        mix_spec: ComplexTensor(B, T, [C,] F)
        ref_spec: List[ComplexTensor(B, T, [C,] F), ...]
        noise_spec: ComplexTensor(B, T, [C,] F)
            only used for IBM and IRM
        mask_type: str
    Returns:
        labels: List[Tensor(B, T, [C,] F), ...] or List[ComplexTensor(B, T, F), ...]
    """

    # Must be upper case
    mask_type = mask_type.upper()
    assert mask_type in [
        "IBM",
        "IRM",
        "IAM",
        "PSM",
        "NPSM",
        "PSM^2",
        "CIRM",
    ], f"mask type {mask_type} not supported"
    mask_label = []
    if ref_spec[0].ndim < mix_spec.ndim:
        # (B, T, F) -> (B, T, 1, F)
        ref_spec = [r.unsqueeze(2).expand_as(mix_spec.real) for r in ref_spec]
    if noise_spec is not None and noise_spec.ndim < mix_spec.ndim:
        # (B, T, F) -> (B, T, 1, F)
        noise_spec = noise_spec.unsqueeze(2).expand_as(mix_spec.real)
    for idx, r in enumerate(ref_spec):
        mask = None
        if mask_type == "IBM":
            if noise_spec is None:
                flags = [abs(r) >= abs(n) for n in ref_spec]
            else:
                flags = [abs(r) >= abs(n) for n in ref_spec + [noise_spec]]
            mask = reduce(lambda x, y: x * y, flags)
            mask = mask.int()
        elif mask_type == "IRM":
            beta = 0.5
            res_spec = sum(n for i, n in enumerate(ref_spec) if i != idx)
            if noise_spec is not None:
                res_spec += noise_spec
            mask = (abs(r).pow(2) / (abs(res_spec).pow(2) + EPS)).pow(beta)
        elif mask_type == "IAM":
            mask = abs(r) / (abs(mix_spec) + EPS)
            mask = mask.clamp(min=0, max=1)
        elif mask_type == "PSM" or mask_type == "NPSM":
            phase_r = r / (abs(r) + EPS)
            phase_mix = mix_spec / (abs(mix_spec) + EPS)
            # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
            cos_theta = phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
            mask = (abs(r) / (abs(mix_spec) + EPS)) * cos_theta
            mask = (
                mask.clamp(min=0, max=1)
                if mask_type == "NPSM"
                else mask.clamp(min=-1, max=1)
            )
        elif mask_type == "PSM^2":
            # This is for training beamforming masks
            phase_r = r / (abs(r) + EPS)
            phase_mix = mix_spec / (abs(mix_spec) + EPS)
            # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
            cos_theta = phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
            mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + EPS)) * cos_theta
            mask = mask.clamp(min=-1, max=1)
        elif mask_type == "CIRM":
            # Ref: Complex Ratio Masking for Monaural Speech Separation
            denominator = mix_spec.real.pow(2) + mix_spec.imag.pow(2) + EPS
            mask_real = (mix_spec.real * r.real + mix_spec.imag * r.imag) / denominator
            mask_imag = (mix_spec.real * r.imag - mix_spec.imag * r.real) / denominator
            mask = new_complex_like(mix_spec, [mask_real, mask_imag])
        assert mask is not None, f"mask type {mask_type} not supported"
        mask_label.append(mask)
    return mask_label


class FrequencyDomainLoss(AbsEnhLoss, ABC):
    """
    Base class for all frequency-domain enhancement loss modules.

    This abstract class provides a structure for defining various types of
    frequency-domain loss functions used in audio enhancement tasks.
    Derived classes must implement specific loss computations and define
    the mask type used for those computations.

    Attributes:
        compute_on_mask (bool): Indicates whether the loss is computed on
            the mask or the spectrum.
        mask_type (str): The type of mask used in loss computation.
        name (str): The name of the loss function.
        only_for_test (bool): If True, this loss is only used during testing.
        is_noise_loss (bool): If True, this loss is related to noise.
        is_dereverb_loss (bool): If True, this loss is related to dereverberation.

    Args:
        name (str): The name of the loss function.
        only_for_test (bool, optional): Whether the loss is only for testing.
            Defaults to False.
        is_noise_loss (bool, optional): Whether the loss is related to noise.
            Defaults to False.
        is_dereverb_loss (bool, optional): Whether the loss is related to
            dereverberation. Defaults to False.

    Raises:
        ValueError: If both `is_noise_loss` and `is_dereverb_loss` are True.

    Examples:
        >>> loss = FrequencyDomainLoss(name="MyLoss", is_noise_loss=True)
        >>> print(loss.name)
        MyLoss
    """

    # The loss will be computed on mask or on spectrum
    @property
    @abstractmethod
    def compute_on_mask() -> bool:
        pass

    # the mask type
    @property
    @abstractmethod
    def mask_type() -> str:
        pass

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
        self, name, only_for_test=False, is_noise_loss=False, is_dereverb_loss=False
    ):
        super().__init__()
        self._name = name
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

    def create_mask_label(self, mix_spec, ref_spec, noise_spec=None):
        """
            Create a mask label based on the provided spectrograms.

        This method generates a mask label for the input mixed spectrogram
        and reference spectrograms based on the specified mask type. It
        utilizes the `_create_mask_label` helper function to compute the
        mask.

        Args:
            mix_spec (ComplexTensor): The mixed spectrogram of shape
                (B, T, [C,] F).
            ref_spec (List[ComplexTensor]): A list of reference spectrograms
                of shape (B, T, [C,] F) for each reference signal.
            noise_spec (ComplexTensor, optional): The noise spectrogram of
                shape (B, T, [C,] F). This is only used for IBM and IRM
                masks. Defaults to None.

        Returns:
            List[Tensor] or List[ComplexTensor]: A list of masks of shape
                (B, T, [C,] F) or (B, T, F) depending on the mask type.

        Examples:
            >>> mix = torch.randn(4, 256, 1, 512)  # Mixed spectrogram
            >>> ref = [torch.randn(4, 256, 1, 512) for _ in range(2)]  # Two refs
            >>> masks = create_mask_label(mix, ref)
            >>> len(masks)  # Should return 2 (one for each reference)

        Note:
            Ensure that the `mask_type` attribute is set correctly in the
            class to use the desired masking method.

        Raises:
            AssertionError: If the `mask_type` is not supported.
        """
        return _create_mask_label(
            mix_spec=mix_spec,
            ref_spec=ref_spec,
            noise_spec=noise_spec,
            mask_type=self.mask_type,
        )


class FrequencyDomainMSE(FrequencyDomainLoss):
    """
        FrequencyDomainMSE computes the Mean Squared Error (MSE) loss in the frequency
    domain for speech enhancement tasks. This loss can be computed either on the
    mask or directly on the spectrum.

    Attributes:
        compute_on_mask (bool): Indicates whether to compute the loss on the mask.
        mask_type (str): The type of mask to be used in the loss calculation.

    Args:
        compute_on_mask (bool): If True, the loss is computed on the mask;
            otherwise, it is computed on the spectrum.
        mask_type (str): The type of mask to use (default is "IBM").
        name (str, optional): Name of the loss instance.
        only_for_test (bool): Indicates if the loss is only for testing (default False).
        is_noise_loss (bool): Indicates if this loss is for noise (default False).
        is_dereverb_loss (bool): Indicates if this loss is for dereverberation (default False).

    Raises:
        ValueError: If `is_noise_loss` and `is_dereverb_loss` are both True.

    Examples:
        >>> loss = FrequencyDomainMSE(compute_on_mask=True, mask_type="IRM")
        >>> ref = torch.randn(2, 100, 256)  # Reference tensor
        >>> inf = torch.randn(2, 100, 256)  # Inferred tensor
        >>> loss_value = loss(ref, inf)  # Compute the MSE loss

        >>> loss = FrequencyDomainMSE()
        >>> ref = torch.randn(2, 100, 1, 256)  # Reference tensor with channels
        >>> inf = torch.randn(2, 100, 1, 256)  # Inferred tensor with channels
        >>> loss_value = loss(ref, inf)  # Compute the MSE loss

    Returns:
        torch.Tensor: The computed MSE loss for each batch.

    Note:
        The input tensors `ref` and `inf` must have the same shape.
    """

    def __init__(
        self,
        compute_on_mask=False,
        mask_type="IBM",
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"MSE_on_{mask_type}"
        else:
            _name = "MSE_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """
            Compute the time-frequency Mean Squared Error (MSE) loss.

        This method calculates the MSE loss between the reference and the
        inferred signals in the frequency domain. The loss is computed
        separately for real and imaginary parts if the inputs are complex.

        Args:
            ref: A tensor of shape (Batch, T, F) or (Batch, T, C, F)
                 representing the reference signal.
            inf: A tensor of shape (Batch, T, F) or (Batch, T, C, F)
                 representing the inferred signal.

        Returns:
            A tensor of shape (Batch,) containing the computed MSE loss
            for each element in the batch.

        Raises:
            ValueError: If the shapes of `ref` and `inf` do not match, or
                        if the dimensions of `ref` are not 3 or 4.

        Examples:
            >>> import torch
            >>> loss_fn = FrequencyDomainMSE()
            >>> ref = torch.rand(8, 100, 256)  # Example reference signal
            >>> inf = torch.rand(8, 100, 256)  # Example inferred signal
            >>> loss = loss_fn.forward(ref, inf)
            >>> print(loss.shape)  # Output: torch.Size([8])
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        diff = ref - inf
        if is_complex(diff):
            mseloss = diff.real**2 + diff.imag**2
        else:
            mseloss = diff**2
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = mseloss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return mseloss


class FrequencyDomainL1(FrequencyDomainLoss):
    """
    Computes the time-frequency L1 loss for audio signal enhancement.

    This class implements the L1 loss function for frequency-domain audio
    enhancement. The L1 loss is computed between the reference signal and the
    estimated signal, which can be either the mask or the spectrum, depending
    on the configuration.

    Attributes:
        compute_on_mask (bool): Indicates whether the loss is computed on
            the mask or the spectrum.
        mask_type (str): The type of mask used for loss computation.

    Args:
        compute_on_mask (bool): If True, computes the loss on the mask;
            otherwise, computes it on the spectrum.
        mask_type (str): The type of mask to be used. Defaults to "IBM".
        name (str): An optional name for the loss instance.
        only_for_test (bool): If True, the loss is only used during testing.
        is_noise_loss (bool): If True, the loss is related to noise.
        is_dereverb_loss (bool): If True, the loss is related to dereverberation.

    Raises:
        ValueError: If both `is_noise_loss` and `is_dereverb_loss` are True.

    Examples:
        >>> import torch
        >>> loss_fn = FrequencyDomainL1(compute_on_mask=True, mask_type="IBM")
        >>> ref = torch.randn(4, 100, 256)  # Example reference signal
        >>> inf = torch.randn(4, 100, 256)  # Example estimated signal
        >>> loss = loss_fn(ref, inf)
        >>> print(loss)  # Outputs the computed L1 loss for the batch

    Note:
        The input tensors `ref` and `inf` should have the same shape,
        which can either be (Batch, T, F) or (Batch, T, C, F).
    """

    def __init__(
        self,
        compute_on_mask=False,
        mask_type="IBM",
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"L1_on_{mask_type}"
        else:
            _name = "L1_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """
            L1 loss in the frequency domain for enhancement tasks.

        This class computes the time-frequency L1 loss between reference and
        estimated signals in the frequency domain. It can operate on either the
        mask or the spectrum, depending on the configuration.

        Attributes:
            compute_on_mask (bool): Indicates whether to compute loss on the mask.
            mask_type (str): The type of mask to be used for loss calculation.

        Args:
            compute_on_mask (bool, optional): If True, compute loss on the mask.
                Defaults to False.
            mask_type (str, optional): Type of mask to use (e.g., "IBM"). Defaults to "IBM".
            name (str, optional): Name of the loss function. Defaults to None.
            only_for_test (bool, optional): If True, this loss is only used for testing.
                Defaults to False.
            is_noise_loss (bool, optional): If True, this loss is used for noise-related
                calculations. Defaults to False.
            is_dereverb_loss (bool, optional): If True, this loss is used for
                dereverberation-related calculations. Defaults to False.

        Returns:
            torch.Tensor: The computed L1 loss.

        Raises:
            ValueError: If the input shapes of `ref` and `inf` do not match or
            if the input shape is invalid.

        Examples:
            >>> import torch
            >>> loss_fn = FrequencyDomainL1()
            >>> ref = torch.randn(10, 2, 4)  # (Batch, T, F)
            >>> inf = torch.randn(10, 2, 4)  # (Batch, T, F)
            >>> loss = loss_fn(ref, inf)
            >>> print(loss.shape)  # (Batch,)

        Note:
            The input tensors `ref` and `inf` should have the same shape.
            The implementation handles both complex and real tensors.
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        if is_complex(inf):
            l1loss = (
                abs(ref.real - inf.real)
                + abs(ref.imag - inf.imag)
                + abs(ref.abs() - inf.abs())
            )
        else:
            l1loss = abs(ref - inf)
        if ref.dim() == 3:
            l1loss = l1loss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = l1loss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss


class FrequencyDomainDPCL(FrequencyDomainLoss):
    """
        FrequencyDomainDPCL is a class that implements the time-frequency Deep Clustering
    loss for audio signal enhancement. This loss function is designed to encourage the
    discriminative separation of audio sources in the frequency domain.

    This class inherits from the FrequencyDomainLoss base class and provides the
    necessary methods and properties to compute the Deep Clustering loss.

    Attributes:
        compute_on_mask (bool): Indicates whether the loss is computed on the mask.
        mask_type (str): The type of mask used for loss computation.
        loss_type (str): Specifies the type of loss, can be "dpcl" or "mdc".

    Args:
        compute_on_mask (bool): Flag to indicate if loss is computed on the mask.
        mask_type (str): Type of mask to be used (default is "IBM").
        loss_type (str): Type of loss to be used ("dpcl" or "mdc", default is "dpcl").
        name (str): Optional name for the loss instance.
        only_for_test (bool): Indicates if the loss is only for testing (default is False).
        is_noise_loss (bool): Indicates if the loss is for noise-related tasks (default is False).
        is_dereverb_loss (bool): Indicates if the loss is for dereverberation tasks (default is False).

    Returns:
        torch.Tensor: A tensor representing the computed loss for the batch.

    Raises:
        ValueError: If an invalid loss type is provided.

    Examples:
        >>> loss = FrequencyDomainDPCL(compute_on_mask=True, mask_type="IRM")
        >>> ref = [torch.rand(2, 100, 256) for _ in range(3)]  # Simulated references
        >>> inf = torch.rand(2, 100 * 256, 3)  # Simulated predictions
        >>> output = loss(ref, inf)
        >>> print(output.shape)  # Should output: torch.Size([2])

    References:
        [1] Deep clustering: Discriminative embeddings for segmentation and
            separation; John R. Hershey. et al., 2016;
            https://ieeexplore.ieee.org/document/7471631
        [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding
            Vectors Based on Regular Simplex; Tanaka, K. et al., 2021;
            https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html
    """

    def __init__(
        self,
        compute_on_mask=False,
        mask_type="IBM",
        loss_type="dpcl",
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "dpcl" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type
        self._loss_type = loss_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """
            Time-frequency Deep Clustering loss.

        This class implements the Deep Clustering loss used for speech separation
        tasks. The loss can be calculated based on either the output embeddings
        or the reference signals. It is designed to work with both "dpcl" and
        "mdc" loss types.

        References:
            [1] Deep clustering: Discriminative embeddings for segmentation and
                separation; John R. Hershey et al., 2016;
                https://ieeexplore.ieee.org/document/7471631
            [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding
                Vectors Based on Regular Simplex; Tanaka, K. et al., 2021;
                https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html

        Args:
            compute_on_mask: If True, compute the loss on the mask instead of
                the spectrum.
            mask_type: The type of mask to be used (default: "IBM").
            loss_type: The type of loss to compute ("dpcl" or "mdc").
            name: Optional name for the loss instance.
            only_for_test: If True, the loss is only used during testing.
            is_noise_loss: If True, this loss is specifically for noise-related
                calculations.
            is_dereverb_loss: If True, this loss is specifically for
                dereverberation-related calculations.

        Returns:
            loss: A tensor containing the computed loss for each batch.
        """
        assert len(ref) > 0
        num_spk = len(ref)

        # Compute the ref for Deep Clustering[1][2]
        abs_ref = [abs(n) for n in ref]
        if self._loss_type == "dpcl":
            r = torch.zeros_like(abs_ref[0])
            B = ref[0].shape[0]
            for i in range(num_spk):
                flags = [abs_ref[i] >= n for n in abs_ref]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int() * i
                r += mask
            r = r.contiguous().flatten().long()
            re = F.one_hot(r, num_classes=num_spk)
            re = re.contiguous().view(B, -1, num_spk)
        elif self._loss_type == "mdc":
            B = ref[0].shape[0]
            manifold_vector = torch.full(
                (num_spk, num_spk),
                (-1 / num_spk) * math.sqrt(num_spk / (num_spk - 1)),
                dtype=inf.dtype,
                device=inf.device,
            )
            for i in range(num_spk):
                manifold_vector[i][i] = ((num_spk - 1) / num_spk) * math.sqrt(
                    num_spk / (num_spk - 1)
                )

            re = torch.zeros(
                ref[0].shape[0],
                ref[0].shape[1],
                ref[0].shape[2],
                num_spk,
                device=inf.device,
            )
            for i in range(num_spk):
                flags = [abs_ref[i] >= n for n in abs_ref]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
                re[mask == 1] = manifold_vector[i]
            re = re.contiguous().view(B, -1, num_spk)
        else:
            raise ValueError(
                f"Invalid loss type error: {self._loss_type}, "
                'the loss type must be "dpcl" or "mdc"'
            )

        V2 = torch.matmul(torch.transpose(inf, 2, 1), inf).pow(2).sum(dim=(1, 2))
        Y2 = (
            torch.matmul(torch.transpose(re, 2, 1).float(), re.float())
            .pow(2)
            .sum(dim=(1, 2))
        )
        VY = torch.matmul(torch.transpose(inf, 2, 1), re.float()).pow(2).sum(dim=(1, 2))

        return V2 + Y2 - 2 * VY


class FrequencyDomainAbsCoherence(FrequencyDomainLoss):
    """
    Computes the absolute coherence loss in the frequency domain.

    This loss is used to measure the coherence between the reference and
    inferred complex tensors in the frequency domain, which is useful in
    tasks such as source separation and enhancement.

    Reference:
        Independent Vector Analysis with Deep Neural Network Source Priors;
        Li et al 2020; https://arxiv.org/abs/2008.11273

    Attributes:
        compute_on_mask (bool): Indicates whether the loss is computed on the
            mask. Always returns False for this class.
        mask_type (str): The type of mask. Always returns None for this class.

    Args:
        compute_on_mask (bool): Whether to compute the loss on the mask.
            Default is False.
        mask_type (str): Type of mask used for the loss computation. Default
            is None.
        name (str): Optional name for the loss instance.
        only_for_test (bool): If True, indicates the loss is only for testing.
            Default is False.
        is_noise_loss (bool): If True, indicates this loss is related to noise.
            Default is False.
        is_dereverb_loss (bool): If True, indicates this loss is related to
            dereverberation. Default is False.

    Returns:
        torch.Tensor: The computed loss for the batch, with shape (Batch,).

    Raises:
        ValueError: If the shapes of `ref` and `inf` do not match or if
        they are not complex tensors.

    Examples:
        >>> loss_fn = FrequencyDomainAbsCoherence()
        >>> ref = torch.randn(8, 128, 64, dtype=torch.complex64)
        >>> inf = torch.randn(8, 128, 64, dtype=torch.complex64)
        >>> loss = loss_fn(ref, inf)
        >>> print(loss.shape)  # Output: torch.Size([8])
    """

    def __init__(
        self,
        compute_on_mask=False,
        mask_type=None,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "Coherence_on_Spec" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = False
        self._mask_type = None

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """
            Computes the time-frequency absolute coherence loss.

        This loss is designed to measure the absolute coherence between the
        reference and the inferred spectrograms. It is particularly useful in
        the context of source separation tasks.

        Reference:
            Independent Vector Analysis with Deep Neural Network Source Priors;
            Li et al 2020; https://arxiv.org/abs/2008.11273

        Attributes:
            compute_on_mask (bool): Indicates if the computation is performed on
                the mask.
            mask_type (str): The type of mask being used (not applicable for
                this class).

        Args:
            compute_on_mask (bool, optional): Flag indicating if the loss is
                computed on the mask. Defaults to False.
            mask_type (str, optional): The type of mask to be used. Defaults to
                None.
            name (str, optional): Name of the loss. Defaults to "Coherence_on_Spec".
            only_for_test (bool, optional): Indicates if the loss is only for
                testing. Defaults to False.
            is_noise_loss (bool, optional): Indicates if the loss is related to
                noise. Defaults to False.
            is_dereverb_loss (bool, optional): Indicates if the loss is related to
                dereverberation. Defaults to False.

        Returns:
            torch.Tensor: The computed loss of shape (Batch,).

        Raises:
            ValueError: If the input tensors do not have the correct dimensions
                or if they are not complex tensors.

        Examples:
            >>> ref = torch.randn(32, 100, 256, dtype=torch.complex64)
            >>> inf = torch.randn(32, 100, 256, dtype=torch.complex64)
            >>> loss_fn = FrequencyDomainAbsCoherence()
            >>> loss = loss_fn(ref, inf)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        if is_complex(ref) and is_complex(inf):
            # sqrt( E[|inf|^2] * E[|ref|^2] )
            denom = (
                complex_norm(ref, dim=1) * complex_norm(inf, dim=1) / ref.size(1) + EPS
            )
            coh = (inf * ref.conj()).mean(dim=1).abs() / denom
            if ref.dim() == 3:
                coh_loss = 1.0 - coh.mean(dim=1)
            elif ref.dim() == 4:
                coh_loss = 1.0 - coh.mean(dim=[1, 2])
            else:
                raise ValueError(
                    "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
                )
        else:
            raise ValueError("`ref` and `inf` must be complex tensors.")
        return coh_loss


class FrequencyDomainCrossEntropy(FrequencyDomainLoss):
    """
        FrequencyDomainCrossEntropy computes the cross-entropy loss in the frequency
    domain for audio signal processing tasks.

    This loss is used to evaluate the difference between the predicted and reference
    distributions in a frequency domain representation, which is useful in
    applications such as speech enhancement and separation.

    Attributes:
        compute_on_mask (bool): Indicates whether the loss is computed on the mask.
        mask_type (str): Type of the mask being used for computation.
        ignore_id (int): ID to ignore in the cross-entropy calculation.

    Args:
        compute_on_mask (bool, optional): If True, the loss is computed on the mask.
            Defaults to False.
        mask_type (str, optional): Type of mask. Defaults to None.
        ignore_id (int, optional): ID to ignore in the loss computation. Defaults to -100.
        name (str, optional): Name of the loss instance. Defaults to None.
        only_for_test (bool, optional): If True, the loss is only for testing. Defaults to False.
        is_noise_loss (bool, optional): If True, the loss is related to noise. Defaults to False.
        is_dereverb_loss (bool, optional): If True, the loss is related to dereverberation.
            Defaults to False.

    Returns:
        torch.Tensor: Computed loss for each batch.

    Examples:
        >>> loss_fn = FrequencyDomainCrossEntropy()
        >>> ref = torch.tensor([[0, 1, 2], [1, 2, -100]])  # Reference labels
        >>> inf = torch.rand(2, 3, 4)  # Predicted logits (Batch, T, nclass)
        >>> loss = loss_fn(ref, inf)
        >>> print(loss)

    Raises:
        ValueError: If the input shapes of `ref` and `inf` do not match or are invalid.

    Note:
        - The input `ref` should be of shape (Batch, T) or (Batch, T, C)
          where T is the time dimension.
        - The input `inf` should be of shape (Batch, T, nclass) or
          (Batch, T, C, nclass).
        - The `ignore_id` can be used to exclude certain labels from the loss
          computation.
    """

    def __init__(
        self,
        compute_on_mask=False,
        mask_type=None,
        ignore_id=-100,
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"CE_on_{mask_type}"
        else:
            _name = "CE_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_id, reduction="none"
        )
        self.ignore_id = ignore_id

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """
            Computes the time-frequency cross-entropy loss for audio enhancement tasks.

        This loss function is designed to compare a reference tensor with an
        inference tensor, both of which represent audio data in a time-frequency
        domain. The cross-entropy loss is useful in tasks such as speech
        separation and enhancement, where the goal is to classify or predict
        the presence of certain audio signals.

        Attributes:
            compute_on_mask (bool): Indicates whether the loss is computed on the
                mask or the spectrum.
            mask_type (str): The type of mask being used (e.g., "IBM", "IRM").
            ignore_id (int): The label ID to ignore during loss computation.

        Args:
            compute_on_mask (bool): If True, the loss will be computed on the
                mask; otherwise, it will be computed on the spectrum.
            mask_type (str, optional): The type of mask to use. Defaults to None.
            ignore_id (int, optional): The label ID to ignore. Defaults to -100.
            name (str, optional): The name of the loss function. Defaults to None.
            only_for_test (bool, optional): If True, indicates the loss is only
                for testing. Defaults to False.
            is_noise_loss (bool, optional): If True, indicates the loss is
                related to noise. Defaults to False.
            is_dereverb_loss (bool, optional): If True, indicates the loss is
                related to dereverberation. Defaults to False.

        Returns:
            torch.Tensor: The computed loss for the batch, shape (Batch,).

        Raises:
            ValueError: If the shapes of `ref` and `inf` are not compatible
                or if the input dimensions are invalid.

        Examples:
            >>> loss_fn = FrequencyDomainCrossEntropy()
            >>> ref = torch.tensor([[1, 0], [0, 1]])  # Reference tensor
            >>> inf = torch.tensor([[[0.1, 0.9], [0.8, 0.2]],
            ...                      [[0.7, 0.3], [0.2, 0.8]]])  # Inference tensor
            >>> loss = loss_fn(ref, inf)  # Compute loss
            >>> print(loss)  # Output the loss value

        Note:
            Ensure that the input tensors `ref` and `inf` have compatible shapes
            as per the specifications mentioned in the Args section.
        """
        assert ref.shape[0] == inf.shape[0] and ref.shape[1] == inf.shape[1], (
            ref.shape,
            inf.shape,
        )

        if ref.dim() == 2:
            loss = self.cross_entropy(inf.permute(0, 2, 1), ref).mean(dim=1)
        elif ref.dim() == 3:
            loss = self.cross_entropy(inf.permute(0, 3, 1, 2), ref).mean(dim=[1, 2])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        with torch.no_grad():
            pred = inf.argmax(-1)
            mask = ref != self.ignore_id
            numerator = (pred == ref).masked_fill(~mask, 0).float()
            if ref.dim() == 2:
                acc = numerator.sum(dim=1) / mask.sum(dim=1).float()
            elif ref.dim() == 3:
                acc = numerator.sum(dim=[1, 2]) / mask.sum(dim=[1, 2]).float()
            self.stats = {"acc": acc.cpu() * 100}

        return loss
