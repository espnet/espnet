# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetDiarizationModel(AbsESPnetModel):
    """Speaker Diarization model"""

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        loss_type: str = "pit",  # only support pit loss for now
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_spk = decoder.num_spk
        self.normalize = normalize
        self.frontend = frontend
        self.label_aggregator = label_aggregator
        self.loss_type = loss_type

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            spk_labels: (Batch, )
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2. Decoder (baiscally a predction layer after encoder_out)
        pred = self.decoder(encoder_out, encoder_out_lens)

        # 3. Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )

        if self.loss_type == "pit":
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )

            (
                correct,
                num_frames,
                speech_scored,
                speech_miss,
                speech_falarm,
                speaker_scored,
                speaker_miss,
                speaker_falarm,
                speaker_error,
            ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

            if speech_scored > 0 and num_frames > 0:
                sad_mr, sad_fr, mi, fa, cf, acc, der = (
                    speech_miss / speech_scored,
                    speech_falarm / speech_scored,
                    speaker_miss / speaker_scored,
                    speaker_falarm / speaker_scored,
                    speaker_error / speaker_scored,
                    correct / num_frames,
                    (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
                )
            else:
                sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0
            stats = dict(
                loss=loss.detach(),
                sad_mr=sad_mr,
                sad_fr=sad_fr,
                mi=mi,
                fa=fa,
                cf=cf,
                acc=acc,
                der=der,
            )
        else:
            raise NotImplementedError

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 3. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
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

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

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

    def pit_loss_single_permute(self, pred, label, length):
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        mask = self.create_length_mask(length, label.size(1), label.size(2))
        loss = bce_loss(pred, label)
        loss = loss * mask
        loss = torch.sum(torch.mean(loss, dim=2), dim=1)
        loss = torch.unsqueeze(loss, dim=1)
        return loss

    def pit_loss(self, pred, label, lengths):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND
        num_output = label.size(2)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        loss_list = []
        for p in permute_list:
            label_perm = label[:, :, p]
            loss_perm = self.pit_loss_single_permute(pred, label_perm, lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss) / torch.sum(lengths.float())
        batch_size = len(min_idx)
        label_list = []
        for i in range(batch_size):
            label_list.append(label[i, :, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length, max_len, num_output):
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        mask = to_device(self, mask)
        return mask

    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )
