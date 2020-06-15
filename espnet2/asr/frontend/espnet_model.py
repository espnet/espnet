from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from itertools import permutations

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
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

    def forward(
            self,
            speech_mix: torch.Tensor,
            speech_ref1: torch.Tensor,
            speech_ref2: torch.Tensor,
            speech_mix_lengths: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples)
            speech_ref: (Batch, num_speaker, samples)
            speech_lengths: (Batch,)
        """
        speech_lengths = speech_mix_lengths
        speech_ref = torch.stack([speech_ref1, speech_ref2], dim=1)
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

        speech_pre, speech_lengths = self.frontend.forward_rawwav(speech_mix, speech_lengths)

        si_snr_loss = self._permutation_loss(torch.unbind(speech_ref, dim=1),
                                             torch.unbind(speech_pre, dim=1), self.si_snr_loss)
        si_snr = - si_snr_loss.detach()
        stats = dict(
            si_snr=si_snr,
        )

        loss = si_snr_loss

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
    def _permutation_loss(ref, inf, criterion):
        """
        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)

        loss, perm = torch.min(losses, dim=1)

        return loss.mean()

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
