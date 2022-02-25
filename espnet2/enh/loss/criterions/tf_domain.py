from abc import ABC
from abc import abstractmethod
from distutils.version import LooseVersion
import torch.nn.functional as F
from functools import reduce

import math
import torch

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


def _create_mask_label(mix_spec, ref_spec, mask_type="IAM"):
    """Create mask label.

    Args:
        mix_spec: ComplexTensor(B, T, [C,] F)
        ref_spec: List[ComplexTensor(B, T, [C,] F), ...]
        mask_type: str
    Returns:
        labels: List[Tensor(B, T, [C,] F), ...] or List[ComplexTensor(B, T, F), ...]
    """

    # Must be upper case
    assert mask_type in [
        "IBM",
        "IRM",
        "IAM",
        "PSM",
        "NPSM",
        "PSM^2",
    ], f"mask type {mask_type} not supported"
    mask_label = []
    for r in ref_spec:
        mask = None
        if mask_type == "IBM":
            flags = [abs(r) >= abs(n) for n in ref_spec]
            mask = reduce(lambda x, y: x * y, flags)
            mask = mask.int()
        elif mask_type == "IRM":
            # TODO(Wangyou): need to fix this,
            #  as noise referecens are provided separately
            mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + EPS)
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
        assert mask is not None, f"mask type {mask_type} not supported"
        mask_label.append(mask)
    return mask_label


class FrequencyDomainLoss(AbsEnhLoss, ABC):

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

    def create_mask_label(self, mix_spec, ref_spec):
        return _create_mask_label(
            mix_spec=mix_spec, ref_spec=ref_spec, mask_type=self.mask_type
        )


class FrequencyDomainMSE(FrequencyDomainLoss):
    def __init__(self, compute_on_mask=False, mask_type="IBM"):
        super().__init__()
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        if self.compute_on_mask:
            return f"MSE_on_{self.mask_type}"
        else:
            return "MSE_on_Spec"

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
            mseloss = diff.real ** 2 + diff.imag ** 2
        else:
            mseloss = diff ** 2
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
    def __init__(self, compute_on_mask=False, mask_type="IBM"):
        super().__init__()
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        if self.compute_on_mask:
            return f"L1_on_{self.mask_type}"
        else:
            return "L1_on_Spec"

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
            l1loss = abs(ref - inf + EPS)
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
    def __init__(self, compute_on_mask=False, mask_type="IBM", loss_type="dpcl"):
        super().__init__()
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type
        self._loss_type = loss_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    @property
    def name(self) -> str:
        return "dpcl"

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency Deep Clustering loss.

        References:
            [1] Deep clustering: Discriminative embeddings for segmentation and separation;
                John R. Hershey. et al., 2016;
                https://ieeexplore.ieee.org/document/7471631
            [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding Vectors Based on Regular Simplex; 
                Tanaka, K. et al., 2021;
                https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html

        Args:
            ref: List[(Batch, T, F) * spks]
            inf: (Batch, T*F, D)
        Returns:
            loss: (Batch,)
        """
        assert len(ref) > 0
        num_spk = len(ref)

        # Compute the ref for Deep Clustering[1][2]
        if self._loss_type == "dpcl":
            r = torch.zeros_like(abs(ref[0]))
            B = ref[0].shape[0]
            for i in range(0, num_spk):
                flags = [abs(ref[i]) >= abs(n) for n in ref]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int() * i
                r += mask
            r = r.contiguous().view(-1,).long()
            re = F.one_hot(r, num_classes=num_spk)
            re = re.contiguous().view(B, -1, num_spk)
        elif self._loss_type == "mdc":
            B = ref[0].shape[0]
            manifold_vector = (
                torch.ones(num_spk, num_spk, device=inf.device)
                * (-1 / num_spk)
                * math.sqrt(num_spk / (num_spk - 1))
            )
            for i in range(num_spk):
                manifold_vector[i][i] = ((num_spk - 1) / num_spk) * math.sqrt(
                    num_spk / (num_spk - 1)
                )

            re = torch.zeros(
                ref[0].shape[0], ref[0].shape[1], ref[0].shape[2], num_spk
            ).to(inf.device)
            for i in range(0, num_spk):
                flags = [abs(ref[i]) >= abs(n) for n in ref]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
                re[mask == 1] = manifold_vector[i]
            re = re.contiguous().view(B, -1, num_spk)
        else:
            raise ValueError(
                'Invalid loss type error: {}, the loss type must be "dpcl" or "mdc"'.format(
                    self._loss_type
                )
            )

        V2 = torch.matmul(torch.transpose(inf, 2, 1), inf).pow(2).sum(dim=(1, 2))
        Y2 = (
            torch.matmul(torch.transpose(re, 2, 1).float(), re.float())
            .pow(2)
            .sum(dim=(1, 2))
        )
        VY = torch.matmul(torch.transpose(inf, 2, 1), re.float()).pow(2).sum(dim=(1, 2))

        return V2 + Y2 - 2 * VY
