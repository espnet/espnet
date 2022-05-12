from abc import ABC
from abc import abstractmethod
from packaging.version import parse as V
from functools import reduce
import math

import torch
import torch.nn.functional as F

from espnet2.enh.layers.complex_utils import complex_norm
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.complex_utils import new_complex_like
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
    """Base class for all frequence-domain Enhancement loss modules."""

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

    def create_mask_label(self, mix_spec, ref_spec, noise_spec=None):
        return _create_mask_label(
            mix_spec=mix_spec,
            ref_spec=ref_spec,
            noise_spec=noise_spec,
            mask_type=self.mask_type,
        )


class FrequencyDomainMSE(FrequencyDomainLoss):
    def __init__(self, compute_on_mask=False, mask_type="IBM", name=None):
        super().__init__()
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

        if name is not None:
            self._name = name
        elif self.compute_on_mask:
            self._name = f"MSE_on_{self.mask_type}"
        else:
            self._name = "MSE_on_Spec"

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        return self._name

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
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
    def __init__(self, compute_on_mask=False, mask_type="IBM", name=None):
        super().__init__()
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

        if name is not None:
            self._name = name
        elif self.compute_on_mask:
            self._name = f"L1_on_{self.mask_type}"
        else:
            self._name = "L1_on_Spec"

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        return self._name

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency L1 loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
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
    def __init__(
        self, compute_on_mask=False, mask_type="IBM", loss_type="dpcl", name=None
    ):
        super().__init__()
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type
        self._loss_type = loss_type
        self._name = "dpcl" if name is None else name

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        return self._name

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency Deep Clustering loss.

        References:
            [1] Deep clustering: Discriminative embeddings for segmentation and
                separation; John R. Hershey. et al., 2016;
                https://ieeexplore.ieee.org/document/7471631
            [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding
                Vectors Based on Regular Simplex; Tanaka, K. et al., 2021;
                https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html

        Args:
            ref: List[(Batch, T, F) * spks]
            inf: (Batch, T*F, D)
        Returns:
            loss: (Batch,)
        """  # noqa: E501
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
    def __init__(self, compute_on_mask=False, mask_type=None, name=None):
        super().__init__()
        self._compute_on_mask = False
        self._mask_type = None

        self._name = "Coherence_on_Spec" if name is None else name

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        return self._name

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency absolute coherence loss.

        Reference:
            Independent Vector Analysis with Deep Neural Network Source Priors;
            Li et al 2020; https://arxiv.org/abs/2008.11273

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
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
    def __init__(self, compute_on_mask=False, mask_type=None, name=None):
        super().__init__()
        self._compute_on_mask = False
        self._mask_type = None

        if name is not None:
            self._name = name
        elif self.compute_on_mask:
            self._name = f"CE_on_{self.mask_type}"
        else:
            self._name = "CE_on_Spec"

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        return self._name

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency cross-entropy loss.

        Args:
            ref: (Batch, T) or (Batch, T, C)
            inf: (Batch, T, nclass) or (Batch, T, C, nclass)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape[0] == inf.shape[0] and ref.shape[1] == inf.shape[1], (
            ref.shape,
            inf.shape,
        )

        if ref.dim() == 2:
            loss = torch.nn.functional.cross_entropy(
                inf.permute(0, 2, 1), ref, reduction="none"
            ).mean(dim=1)
        elif ref.dim() == 3:
            loss = torch.nn.functional.cross_entropy(
                inf.permute(0, 3, 1, 2), ref, reduction="none"
            ).mean(dim=[1, 2])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        with torch.no_grad():
            pred = inf.argmax(-1)
            acc = (pred == ref).float()
            if ref.dim() == 2:
                acc = acc.mean(dim=1)
            elif ref.dim() == 3:
                acc = acc.mean(dim=[1, 2])
            self.stats = {"acc": acc.cpu() * 100}

        return loss
