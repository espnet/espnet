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
        speech_ref = torch.stack([speech_ref1, speech_ref2], dim = 1)
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

        # 1. Encoder
        predict_magnitude, feats_lens = self.frontend(speech_mix, speech_lengths)

        ref_stft = [self.frontend.wav_to_stft(s_spk, speech_lengths)[0] for s_spk in torch.unbind(speech_ref, dim=1)]

        ref_stft = [stft[:, : max(feats_lens), :] for stft in ref_stft]
        ref_magnitude = [abs(stft) for stft in ref_stft]
        
        loss = self._cal_permutation_loss(ref_magnitude, torch.unbind(predict_magnitude, dim=1))

        stats = dict(
            loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _cal_permutation_loss(
        self, 
        ref_speechs: List[torch.Tensor],
        inf_speechs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            ref_speechs (List[torch.Tensor]): [(batch, T, F), ...]
            inf_speechs (List[torch.Tensor]): [(batch, T, F), ...]

        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref_speechs)

        criterion = torch.nn.L1Loss(reduction='none')


        def lossfunc(permutation):
            return sum(
                [criterion(ref_speechs[s], inf_speechs[t]).mean(dim=[1,2]) 
                for s, t in enumerate(permutation)]
                 ) / len(permutation)

        losses = torch.stack([lossfunc(p) for p in permutations(range(num_spk))], dim=1)

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




