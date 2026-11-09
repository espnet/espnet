# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.abs_look_ahead_encoder import AbsLookAheadEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.legacy.nets.pytorch_backend.nets_utils import to_device
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetDiarizationModel(AbsESPnetModel):
    """Speaker Diarization model

    If "attractor" is "None", SA-EEND will be used.
    Else if "attractor" is not "None", EEND-EDA will be used.
    For the details about SA-EEND and EEND-EDA, refer to the following papers:
    SA-EEND: https://arxiv.org/pdf/1909.06247.pdf
    EEND-EDA: https://arxiv.org/pdf/2005.09921.pdf, https://arxiv.org/pdf/2106.10654.pdf
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        attractor: Optional[AbsAttractor],
        diar_weight: float = 1.0,
        attractor_weight: float = 1.0,
        look_aheads: Optional[List[int]] = None,
        wav_augment: Optional[torch.nn.Module] = None,
    ):

        super().__init__()

        # waveform augmentation of the batch, training only (e.g. BatchWavAugment)
        self.wav_augment = wav_augment

        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.label_aggregator = label_aggregator
        self.diar_weight = diar_weight
        self.attractor_weight = attractor_weight
        self.attractor = attractor
        self.decoder = decoder

        if self.attractor is not None:
            self.decoder = None
        elif self.decoder is not None:
            self.num_spk = decoder.num_spk
        else:
            raise NotImplementedError

        # Look-aheads (input frames) of the encoder that each forward pass
        # trains and validates; the loss is their mean (dual-mode training).
        self.look_aheads = list(look_aheads) if look_aheads else []
        if self.look_aheads:
            if not isinstance(encoder, AbsLookAheadEncoder):
                raise ValueError(
                    f"look_aheads needs an encoder with a look-ahead, "
                    f"not {type(encoder).__name__}"
                )
            start = encoder.look_ahead
            try:
                for frames in self.look_aheads:
                    encoder.set_look_ahead(frames)  # raises for an invalid one
            finally:
                encoder.set_look_ahead(start)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
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
            kwargs: "utt_id" is among the input.
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        if self.wav_augment is not None:
            # The module augments in training mode only; in eval mode it only
            # removes its leading context, if it has one.
            speech, speech_lengths = self.wav_augment(speech, speech_lengths)

        if not self.look_aheads:
            loss, stats = self._forward_one(
                speech, speech_lengths, spk_labels, spk_labels_lengths, **kwargs
            )
        else:
            loss, stats = self._forward_look_aheads(
                speech, speech_lengths, spk_labels, spk_labels_lengths, **kwargs
            )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def set_look_ahead(self, frames: int):
        """Make the encoder read up to `frames` future frames (0: online).

        Raises:
            ValueError: The encoder is not an AbsLookAheadEncoder, or it cannot
                use this look-ahead.
        """
        if not isinstance(self.encoder, AbsLookAheadEncoder):
            raise ValueError(
                f"look_ahead needs an encoder with a look-ahead, "
                f"not {type(self.encoder).__name__}"
            )
        self.encoder.set_look_ahead(frames)

    def _forward_look_aheads(
        self, speech, speech_lengths, spk_labels, spk_labels_lengths, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Run _forward_one at each look-ahead and average the losses.

        The statistics of each look-ahead get the suffix "_la<frames>"; the
        plain names hold their mean. The encoder gets its look-ahead back.
        """
        start = self.encoder.look_ahead
        losses, per_look_ahead = [], []
        try:
            for frames in self.look_aheads:
                self.encoder.set_look_ahead(frames)
                # A copy per look-ahead: without a frontend, the normalization
                # may work on the input in place (GlobalMVN without gradient).
                loss, stats = self._forward_one(
                    speech.clone(),
                    speech_lengths,
                    spk_labels,
                    spk_labels_lengths,
                    **kwargs,
                )
                losses.append(loss)
                per_look_ahead.append((frames, stats))
        finally:
            self.encoder.set_look_ahead(start)
        loss = torch.stack(losses).mean()
        stats = {}
        for frames, one in per_look_ahead:
            for key, value in one.items():
                if value is not None:
                    stats[f"{key}_la{frames}"] = value
        for key in per_look_ahead[0][1]:
            values = [one[key] for _, one in per_look_ahead if one[key] is not None]
            stats[key] = torch.stack(values).mean() if values else None
        stats["loss"] = loss.detach()
        return loss, stats

    def _forward_one(
        self, speech, speech_lengths, spk_labels, spk_labels_lengths, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Return the loss and the statistics at the encoder's look-ahead."""
        # 1. Encoder
        # Use bottleneck_feats if exist. Only for "enh + diar" task.
        bottleneck_feats = kwargs.get("bottleneck_feats", None)
        bottleneck_feats_lengths = kwargs.get("bottleneck_feats_lengths", None)
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths
        )

        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                    i, torch.randperm(encoder_out_lens[i]), :
                ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))
        # 3. Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[1] - pred.shape[1]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, 0 : pred.shape[1], :]

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
        else:
            loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
            loss_att = self.attractor_loss(att_prob, spk_labels)
            loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
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

        # Ratios on the device (0 when nothing is scored), so the forward pass
        # never waits for the GPU; the reporter reads them after backward.
        scored = (speech_scored > 0) & (num_frames > 0)

        def ratio(num, den):
            return torch.where(scored, num / den.clamp(min=1), torch.zeros_like(num))

        sad_mr = ratio(speech_miss, speech_scored)
        sad_fr = ratio(speech_falarm, speech_scored)
        mi = ratio(speaker_miss, speaker_scored)
        fa = ratio(speaker_falarm, speaker_scored)
        cf = ratio(speaker_error, speaker_scored)
        acc = ratio(correct, num_frames)
        der = ratio(speaker_miss + speaker_falarm + speaker_error, speaker_scored)

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )
        return loss, stats

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.wav_augment is not None and hasattr(self.wav_augment, "remove_context"):
            # The statistics are those of the windows the model sees: without
            # their leading context (and without random augmentation).
            speech, speech_lengths = self.wav_augment.remove_context(
                speech, speech_lengths
            )
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bottleneck_feats: torch.Tensor,
        bottleneck_feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            bottleneck_feats: (Batch, Length, ...): used for enh + diar
        """
        with autocast("cuda", enabled=False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            if bottleneck_feats is None:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            elif self.frontend is None:
                # use only bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    bottleneck_feats, bottleneck_feats_lengths
                )
            else:
                # use both frontend and bottleneck feats
                # interpolate (copy) feats frames
                # to match the length with bottleneck_feats
                feats = F.interpolate(
                    feats.transpose(1, 2), size=bottleneck_feats.shape[1]
                ).transpose(1, 2)
                # concatenate frontend LMF feature and bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    torch.cat((bottleneck_feats, feats), 2), bottleneck_feats_lengths
                )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        # Checked on the device: reading encoder_out_lens.max() here would make
        # the CPU wait for the frontend and encoder before queuing the rest.
        torch._assert_async(encoder_out_lens.max() >= encoder_out.size(1))

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
        # Reorder each sample's label by its best permutation on the device,
        # without a per-sample copy to the CPU.
        permutation = torch.as_tensor(
            np.stack(permute_list), dtype=torch.long, device=label.device
        )[min_idx]
        label_permute = torch.gather(
            label, 2, permutation.unsqueeze(1).expand(-1, label.size(1), -1)
        ).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length, max_len, num_output):
        """Float mask (Batch, max_len, num_output): 1 inside each length, else 0."""
        valid = torch.arange(max_len, device=length.device) < length.unsqueeze(1)
        mask = valid.unsqueeze(2).expand(-1, -1, num_output).float()
        mask = to_device(self, mask)
        return mask

    def attractor_loss(self, att_prob, label):
        batch_size = len(label)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # create attractor label [1, 1, ..., 1, 0]
        # att_label: (Batch, num_spk + 1, 1)
        att_label = to_device(self, torch.zeros(batch_size, label.size(2) + 1, 1))
        att_label[:, : label.size(2), :] = 1
        loss = bce_loss(att_prob, att_label)
        loss = torch.mean(torch.mean(loss, dim=1))
        return loss

    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        # Computed on the device of `pred`, without a copy to the CPU.
        batch_size, max_len, num_output = label.size()
        label = label.to(pred.device)
        length = length.to(pred.device)
        # mask the padding part
        mask = (
            (torch.arange(max_len, device=pred.device) < length.unsqueeze(1))
            .unsqueeze(2)
            .long()
        )

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.long() * mask
        pred_np = (pred > 0).long() * mask

        # compute speech activity detection error
        n_ref = label_np.sum(dim=2)
        n_sys = pred_np.sum(dim=2)
        speech_scored = (n_ref > 0).sum()
        speech_miss = torch.logical_and(n_ref > 0, n_sys == 0).sum()
        speech_falarm = torch.logical_and(n_ref == 0, n_sys > 0).sum()

        # compute speaker diarization error
        speaker_scored = n_ref.sum()
        speaker_miss = torch.clamp(n_ref - n_sys, min=0).sum()
        speaker_falarm = torch.clamp(n_sys - n_ref, min=0).sum()
        n_map = torch.logical_and(label_np == 1, pred_np == 1).sum(dim=2)
        speaker_error = (torch.minimum(n_ref, n_sys) - n_map).sum()
        correct = ((label_np == pred_np) * mask).sum()
        num_frames = length.sum()

        # Float tensors on the device of `pred`: no copy to the CPU here.
        return (
            correct.float() / num_output,
            num_frames.float(),
            speech_scored.float(),
            speech_miss.float(),
            speech_falarm.float(),
            speaker_scored.float(),
            speaker_miss.float(),
            speaker_falarm.float(),
            speaker_error.float(),
        )
