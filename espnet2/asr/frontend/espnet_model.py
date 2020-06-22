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

        # prepare reference speech and reference magnitude
        speech_ref = torch.unbind(speech_ref, dim=1)
        magnitude_ref = [self.frontend.stft(sr)[0] for sr in speech_ref]
        magnitude_ref = [abs(ComplexTensor(mr[..., 0], mr[..., 1])) for mr in magnitude_ref]

        # predict separated speech and separated magnitude
        speech_pre, speech_lengths = self.frontend.forward_rawwav(speech_mix, speech_lengths)
        magnitude_pre, tf_length = self.frontend(speech_mix, speech_lengths)
        magnitude_pre = torch.unbind(magnitude_pre, dim=1)
        speech_pre = torch.unbind(speech_pre, dim=1)

        # compute TF masking loss
        tf_loss, perm = self._permutation_loss(magnitude_ref, magnitude_pre, self.tf_l1_loss)

        # compute si-snr loss
        si_snr_loss, perm = self._permutation_loss(speech_ref, speech_pre, self.si_snr_loss, perm=perm)

        si_snr = - si_snr_loss
        loss = (1 - self.tf_factor) * si_snr_loss + self.tf_factor * tf_loss

        stats = dict(
            si_snr=si_snr.detach(),
            tf_loss=tf_loss.detach(),
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

        if perm == None:
            losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)
            loss, perm = torch.min(losses, dim=1)
        else:
            losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)
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
