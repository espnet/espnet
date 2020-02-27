from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_e2e import AbsE2E


class ASRHybridE2E(AbsE2E):
    """DNN-HMM hybrid model"""

    def __init__(
        self,
        num_targets: int,
        frontend: Optional[AbsFrontend],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.num_targets = num_targets
        self.ignore_id = ignore_id

        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = torch.nn.Linear(encoder.output_size(), num_targets)
        self.criterion = LabelSmoothingLoss(
            size=num_targets,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
        self,
        speech: torch.Tensor,
        align: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        align_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            align: (Batch, Length)
            align_lengths: (Batch,)
        """
        if speech_lengths is None:
            speech_lengths = speech.new_full(
                [speech.shape[0]], fill_value=speech.shape[1],
            )
        if align_lengths is None:
            align_lengths = align.new_full([align.shape[0]], fill_value=align.shape[1],)
        # Check that batch_size is unified
        assert speech.shape[0] == speech_lengths.shape[0] == align.shape[0], (
            speech.shape,
            speech_lengths.shape,
            align.shape,
        )
        batch_size = speech.shape[0]

        if align_lengths is not None:
            # for data-parallel
            align = align[:, : align_lengths.max()]

        # 1. Forward
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # The lenght between input feats and alignment can be differed with 1 frame
        # by the padding strategy of each STFT,
        # then padding or truncating alignment here.
        if encoder_out.shape[1] > align.shape[1]:
            align = torch.nn.functional.pad(
                align, (0, encoder_out.shape[1] - align.shape[1]),
            )
        elif encoder_out.shape[1] < align.shape[1]:
            align = align[:, : encoder_out.shape[1]]
        for l1, l2 in zip(encoder_out_lens, align_lengths):
            if abs(l1 - l2) >= 2:
                raise RuntimeError(
                    f"num-frame is differed by 2 or more frames: "
                    f"speech={l1}, align={l2}"
                )
            if l1 > l2:
                align[l2:l1] = align[l2]
            elif l1 < l2:
                align[l1:l2] = self.ignore_id

        decoder_out = self.decoder(encoder_out)

        # 2. Calc loss
        loss = self.criterion(decoder_out, align)
        acc = th_accuracy(
            decoder_out.view(-1, self.num_targets), align, ignore_label=self.ignore_id,
        )

        stats = dict(loss=loss.detach(), acc=acc)

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        align: torch.Tensor,
        align_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_decode.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 3. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def nnet_forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        decoder_out = self.decoder(encoder_out)
        decoder_out.masked_fill_(make_pad_mask(encoder_out_lens, decoder_out, 1), 0.0)
        return decoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths
