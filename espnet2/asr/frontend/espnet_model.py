from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from collections import OrderedDict
from itertools import permutations

import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from torch_complex.tensor import ComplexTensor
from functools import reduce


class ESPnetFrontendModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            frontend: Optional[AbsFrontend],
    ):
        assert check_argument_types()

        super().__init__()

        self.frontend = frontend
        self.num_spk = frontend.num_spk
        self.fs = frontend.fs
        self.tf_factor = frontend.tf_factor
        self.mask_type = frontend.mask_type

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
        """
        :param mix_spec: ComplexTensor(B, T, F)
        :param ref_spec: [ComplexTensor(B, T, F), ...] or ComplexTensor(B, T, F)
        :param noise_spec: ComplexTensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

        assert mask_type in ["IBM", "IRM", "IAM", "PSM", "NPSM", "ICM"], f"mask type {mask_type} not supported"
        eps = 10 - 8
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + eps)
            elif mask_type == "IAM":
                mask = (abs(r)) / (abs(mix_spec) + eps)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                cos_theta = phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                mask = (abs(r) / (abs(mix_spec) + eps)) * cos_theta
                mask = mask.clamp(min=0, max=1) if mask_label == "NPSM" else mask.clamp(min=-1, max=1)
            elif mask_type == "ICM":
                mask = r / (mix_spec + eps)
                mask.real = mask.real.clamp(min=-1, max=1)
                mask.imag = mask.imag.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
            self,
            speech_mix: torch.Tensor,
            speech_mix_lengths: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples)
            speech_ref: (Batch, num_speaker, samples)
            speech_lengths: (Batch,)
        """
        # (Batch, num_speaker, samples)
        speech_ref = torch.stack([
            kwargs['speech_ref{}'.format(spk + 1)] for spk in range(self.num_spk)
        ], dim=1)
        speech_lengths = speech_mix_lengths
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert (
                speech_mix.shape[0]
                == speech_ref.shape[0]
                == speech_lengths.shape[0]
        ), (speech_mix.shape, speech_ref.shape, speech_lengths.shape)
        batch_size = speech_mix.shape[0]

        # for data-parallel
        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        if self.tf_factor:
            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            sepctrum_ref = [self.frontend.stft(sr)[0] for sr in speech_ref]
            sepctrum_ref = [ComplexTensor(sr[..., 0], sr[..., 1]) for sr in sepctrum_ref]
            sepctrum_mix = self.frontend.stft(speech_mix)[0]
            sepctrum_mix = ComplexTensor(sepctrum_mix[..., 0], sepctrum_mix[..., 1])

            # prepare ideal masks
            mask_ref = self._create_mask_label(sepctrum_mix, sepctrum_ref, mask_type=self.mask_type)

            # predict separated speech and separated magnitude
            spectrum_pre, tf_length, mask_pre = self.frontend(speech_mix, speech_lengths)

            # compute TF masking loss
            tf_loss, perm = self._permutation_loss(mask_ref, mask_pre, self.tf_l1_loss)

            speech_pre = [self.frontend.stft.inverse(ps, speech_lengths)[0] for ps in spectrum_pre]

            # compute si-snr loss
            si_snr_loss, perm = self._permutation_loss(speech_ref, speech_pre, self.si_snr_loss, perm=perm)

            si_snr = - si_snr_loss
            loss = (1 - self.tf_factor) * si_snr_loss + self.tf_factor * tf_loss
            stats = dict(
                si_snr=si_snr.detach(),
                tf_loss=tf_loss.detach(),
                loss=loss.detach()
            )
        else:
            # TODO:Jing, should find better way to configure for the choice of tf loss and time-only loss.
            speech_pre, speech_lengths = self.frontend.forward_rawwav(speech_mix, speech_lengths)
            speech_pre = torch.unbind(speech_pre, dim=1)

            # compute si-snr loss
            si_snr_loss, perm = self._permutation_loss(speech_ref, speech_pre, self.si_snr_loss)
            si_snr = - si_snr_loss
            loss = si_snr_loss
            stats = dict(
                si_snr=si_snr.detach(),
                loss=loss.detach()
            )

        loss = si_snr_loss

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @staticmethod
    def tf_l1_loss(ref, inf):
        """
        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        l1loss = abs(ref - inf).mean(dim=[1, 2])

        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """
        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1))
        return -si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """
        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)
        if perm == None:
            loss, perm = torch.min(losses, dim=1)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]
            pass

        return loss.mean(), perm

    def collect_feats(
            self,
            speech_mix: torch.Tensor,
            speech_mix_lengths: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
