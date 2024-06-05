# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based Singing-voice-synthesis ESPnet model."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.gan_svs.abs_gan_svs import AbsGANSVS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.svs.feats_extract.score_feats_extract import (
    FrameScoreFeats,
    SyllableScoreFeats,
    expand_to_frame,
)
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch < 1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetGANSVSModel(AbsGANESPnetModel):
    """ESPnet model for GAN-based singing voice synthesis task."""

    @typechecked
    def __init__(
        self,
        postfrontend: Optional[AbsFrontend],
        text_extract: Optional[AbsFeatsExtract],
        feats_extract: Optional[AbsFeatsExtract],
        score_feats_extract: Optional[AbsFeatsExtract],
        label_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        ying_extract: Optional[AbsFeatsExtract],
        duration_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        svs: AbsGANSVS,
    ):
        """Initialize ESPnetGANSVSModel module."""
        super().__init__()
        self.text_extract = text_extract
        self.feats_extract = feats_extract
        self.score_feats_extract = score_feats_extract
        self.label_extract = label_extract
        self.pitch_extract = pitch_extract
        self.duration_extract = duration_extract
        self.energy_extract = energy_extract
        self.ying_extract = ying_extract
        self.normalize = normalize
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.svs = svs
        self.postfrontend = postfrontend
        assert hasattr(
            svs, "generator"
        ), "generator module must be registered as svs.generator"
        assert hasattr(
            svs, "discriminator"
        ), "discriminator module must be registered as svs.discriminator"

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration_phn: Optional[torch.Tensor] = None,
        duration_phn_lengths: Optional[torch.Tensor] = None,
        duration_ruled_phn: Optional[torch.Tensor] = None,
        duration_ruled_phn_lengths: Optional[torch.Tensor] = None,
        duration_syb: Optional[torch.Tensor] = None,
        duration_syb_lengths: Optional[torch.Tensor] = None,
        slur: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        ying: Optional[torch.Tensor] = None,
        ying_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return generator or discriminator loss with dict format.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            label (Option[Tensor]): Label tensor (B, T_label).
            label_lengths (Optional[Tensor]): Label lrngth tensor (B,).
            phn_cnt (Optional[Tensor]): Number of phones in each syllable (B, T_syb)
            midi (Option[Tensor]): Midi tensor (B, T_label).
            midi_lengths (Optional[Tensor]): Midi lrngth tensor (B,).
            duration_phn (Optional[Tensor]): duration tensor (B, T_label).
            duration_phn_lengths (Optional[Tensor]): duration length tensor (B,).
            duration_ruled_phn (Optional[Tensor]): duration tensor (B, T_phone).
            duration_ruled_phn_lengths (Optional[Tensor]): duration length tensor (B,).
            duration_syb (Optional[Tensor]): duration tensor (B, T_syllable).
            duration_syb_lengths (Optional[Tensor]): duration length tensor (B,).
            slur (Optional[Tensor]): slur tensor (B, T_slur).
            pitch (Optional[Tensor]): Pitch tensor (B, T_wav). - f0 sequence
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            forward_generator (bool): Whether to forward generator.
            kwargs: "utt_id" is among the input.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        with autocast(False):
            # Extract features
            if self.feats_extract is not None and feats is None:
                feats, feats_lengths = self.feats_extract(
                    singing,
                    singing_lengths,
                )
            # Extract auxiliary features
            # score : 128 midi pitch
            # duration :
            #   input-> phone-id seqence
            #   output -> frame level(take mode from window) or syllable level

            # cut length
            for i in range(feats.size(0)):
                dur_len = sum(duration_phn[i])
                if feats_lengths[i] > dur_len:
                    feats_lengths[i] = dur_len
                else:  # decrease duration at the end of sequence
                    delta = dur_len - feats_lengths[i]
                    end = duration_phn_lengths[i] - 1
                    while delta > 0 and end >= 0:
                        new = duration_phn[i][end] - delta
                        if new < 0:  # keep on decreasing the previous one
                            delta -= duration_phn[i][end]
                            duration_phn[i][end] = 0
                            end -= 1
                        else:  # stop
                            delta -= duration_phn[i][end] - new
                            duration_phn[i][end] = new
            feats = feats[:, : feats_lengths.max()]

            if isinstance(self.score_feats_extract, FrameScoreFeats):
                (
                    label_lab,
                    label_lab_lengths,
                    midi_lab,
                    midi_lab_lengths,
                    duration_lab,
                    duration_lab_lengths,
                ) = expand_to_frame(
                    duration_phn, duration_phn_lengths, label, midi, duration_phn
                )

                # for data-parallel
                label_lab = label_lab[:, : label_lab_lengths.max()]
                midi_lab = midi_lab[:, : midi_lab_lengths.max()]
                duration_lab = duration_lab[:, : duration_lab_lengths.max()]

                (
                    label_score,
                    label_score_lengths,
                    midi_score,
                    midi_score_lengths,
                    duration_score,
                    duration_score_phn_lengths,
                ) = expand_to_frame(
                    duration_ruled_phn,
                    duration_ruled_phn_lengths,
                    label,
                    midi,
                    duration_ruled_phn,
                )

                # for data-parallel
                label_score = label_score[:, : label_score_lengths.max()]
                midi_score = midi_score[:, : midi_score_lengths.max()]
                duration_score = duration_score[:, : duration_score_phn_lengths.max()]
                duration_score_syb = None

            elif isinstance(self.score_feats_extract, SyllableScoreFeats):
                label_lab_lengths = label_lengths
                midi_lab_lengths = midi_lengths
                duration_lab_lengths = duration_phn_lengths

                label_lab = label[:, : label_lab_lengths.max()]
                midi_lab = midi[:, : midi_lab_lengths.max()]
                duration_lab = duration_phn[:, : duration_lab_lengths.max()]

                label_score_lengths = label_lengths
                midi_score_lengths = midi_lengths
                duration_score_phn_lengths = duration_ruled_phn_lengths
                duration_score_syb_lengths = duration_syb_lengths

                label_score = label[:, : label_score_lengths.max()]
                midi_score = midi[:, : midi_score_lengths.max()]
                duration_score = duration_ruled_phn[
                    :, : duration_score_phn_lengths.max()
                ]
                duration_score_syb = duration_syb[:, : duration_score_syb_lengths.max()]
                slur = slur[:, : label_score_lengths.max()]

            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    input=singing,
                    input_lengths=singing_lengths,
                    feats_lengths=feats_lengths,
                )

            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    singing,
                    singing_lengths,
                    feats_lengths=feats_lengths,
                )

            if self.ying_extract is not None and ying is None:
                ying, ying_lengths = self.ying_extract(
                    singing,
                    singing_lengths,
                    feats_lengths=feats_lengths,
                )

            if self.postfrontend is not None:
                # extract features using pretrained SSL models like HuBERT
                ssl_feats, ssl_feats_lengths = self.postfrontend(
                    singing, singing_lengths
                )
            else:
                ssl_feats, ssl_feats_lengths = None, None

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for svs inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
            forward_generator=forward_generator,
        )

        # label
        # NOTE(Yuning): Label can be word, syllable or phoneme,
        # which is determined by annotation file.
        label = dict()
        label_lengths = dict()
        if label_lab is not None:
            label_lab = label_lab.to(dtype=torch.long)
            label.update(lab=label_lab)
            label_lengths.update(lab=label_lab_lengths)
        if label_score is not None:
            label_score = label_score.to(dtype=torch.long)
            label.update(score=label_score)
            label_lengths.update(score=label_score_lengths)
        batch.update(label=label, label_lengths=label_lengths)

        # melody
        melody = dict()
        if midi_lab is not None:
            midi_lab = midi_lab.to(dtype=torch.long)
            melody.update(lab=midi_lab)
        if midi_score is not None:
            midi_score = midi_score.to(dtype=torch.long)
            melody.update(score=midi_score)
        batch.update(melody=melody)

        # duration
        # NOTE(Yuning): duration = duration_time / time_shift (same as Xiaoice paper)
        duration = dict()
        if duration_lab is not None:
            duration_lab = duration_lab.to(dtype=torch.long)
            duration.update(lab=duration_lab)
        if duration_score is not None:
            duration_phn_score = duration_score.to(dtype=torch.long)
            duration.update(score_phn=duration_phn_score)
        if duration_score_syb is not None:
            duration_syb_score = duration_score_syb.to(dtype=torch.long)
            duration.update(score_syb=duration_syb_score)
        batch.update(duration=duration)

        if slur is not None:
            batch.update(slur=slur)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if feats is not None:
            batch.update(feats=feats, feats_lengths=feats_lengths)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy)
        if self.ying_extract is not None and ying is not None:
            batch.update(ying=ying)
        if self.svs.require_raw_singing:
            batch.update(singing=singing, singing_lengths=singing_lengths)
        if self.postfrontend is not None:
            batch.update(ssl_feats=ssl_feats, ssl_feats_lengths=ssl_feats_lengths)
        else:
            batch.update(ssl_feats=None, ssl_feats_lengths=None)
        return self.svs(**batch)

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        duration_phn: Optional[torch.Tensor] = None,
        duration_phn_lengths: Optional[torch.Tensor] = None,
        duration_ruled_phn: Optional[torch.Tensor] = None,
        duration_ruled_phn_lengths: Optional[torch.Tensor] = None,
        duration_syb: Optional[torch.Tensor] = None,
        duration_syb_lengths: Optional[torch.Tensor] = None,
        slur: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        ying: Optional[torch.Tensor] = None,
        ying_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Calculate features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            label (Option[Tensor]): Label tensor (B, T_label).
            label_lengths (Optional[Tensor]): Label lrngth tensor (B,).
            phn_cnt (Optional[Tensor]): Number of phones in each syllable (B, T_syb)
            midi (Option[Tensor]): Midi tensor (B, T_label).
            midi_lengths (Optional[Tensor]): Midi lrngth tensor (B,).
            duration_phn (Optional[Tensor]): duration tensor (T_label).
            duration_ruled_phn (Optional[Tensor]): duration tensor (T_phone).
            duration_syb (Optional[Tensor]): duration tensor (T_phone).
            slur (Optional[Tensor]): slur tensor (B, T_slur).
            pitch (Optional[Tensor]): Pitch tensor (B, T_wav). - f0 sequence
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).

        Returns:
            Dict[str, Tensor]: Dict of features.
        """
        feats = None
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(
                singing,
                singing_lengths,
            )

        # cut length
        for i in range(feats.size(0)):
            dur_len = sum(duration_phn[i])
            if feats_lengths[i] > dur_len:
                feats_lengths[i] = dur_len
            else:  # decrease duration at the end of sequence
                delta = dur_len - feats_lengths[i]
                end = duration_phn_lengths[i] - 1
                while delta > 0 and end >= 0:
                    new = duration_phn[i][end] - delta
                    if new < 0:  # keep on decreasing the previous one
                        delta -= duration_phn[i][end]
                        duration_phn[i][end] = 0
                        end -= 1
                    else:  # stop
                        delta -= duration_phn[i][end] - new
                        duration_phn[i][end] = new
        feats = feats[:, : feats_lengths.max()]

        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                input=singing,
                input_lengths=singing_lengths,
                feats_lengths=feats_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                singing,
                singing_lengths,
                feats_lengths=feats_lengths,
            )
        if self.ying_extract is not None and ying is None:
            ying, ying_lengths = self.ying_extract(
                singing,
                singing_lengths,
                feats_lengths=feats_lengths,
            )

        # store in dict
        feats_dict = {}
        if feats is not None:
            feats_dict.update(feats=feats, feats_lengths=feats_lengths)
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)
        if ying is not None:
            feats_dict.update(ying=ying, ying_lengths=ying_lengths)

        return feats_dict

    def inference(
        self,
        text: torch.Tensor,
        singing: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        duration_phn: Optional[torch.Tensor] = None,
        duration_ruled_phn: Optional[torch.Tensor] = None,
        duration_syb: Optional[torch.Tensor] = None,
        slur: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (T_text).
            singing (Tensor): Singing waveform tensor (T_wav).
            label (Option[Tensor]): Label tensor (T_label).
            phn_cnt (Optional[Tensor]): Number of phones in each syllable (T_syb)
            midi (Option[Tensor]): Midi tensor (T_l abel).
            duration_phn (Optional[Tensor]): duration tensor (T_label).
            duration_ruled_phn (Optional[Tensor]): duration tensor (T_phone).
            duration_syb (Optional[Tensor]): duration tensor (T_phone).
            slur (Optional[Tensor]): slur tensor (T_phone).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            pitch (Optional[Tensor): Pitch tensor (T_wav).
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.
        """
        label_lengths = torch.tensor([len(label)])
        midi_lengths = torch.tensor([len(midi)])
        duration_phn_lengths = torch.tensor([len(duration_phn)])
        duration_ruled_phn_lengths = torch.tensor([len(duration_ruled_phn)])
        duration_syb_lengths = torch.tensor([len(duration_syb)])
        slur_lengths = torch.tensor([len(slur)])

        # unsqueeze of singing needed otherwise causing error in STFT dimension
        # for data-parallel
        text = text.unsqueeze(0)

        label = label.unsqueeze(0)
        midi = midi.unsqueeze(0)
        duration_phn = duration_phn.unsqueeze(0)
        duration_ruled_phn = duration_ruled_phn.unsqueeze(0)
        duration_syb = duration_syb.unsqueeze(0)
        phn_cnt = phn_cnt.unsqueeze(0)
        slur = slur.unsqueeze(0)

        # Extract auxiliary features
        # melody : 128 midi pitch
        # duration :
        #   input-> phone-id seqence
        #   output -> frame level or syllable level
        batch_size = text.size(0)
        assert batch_size == 1
        if isinstance(self.score_feats_extract, FrameScoreFeats):
            (
                label_lab,
                label_lab_lengths,
                midi_lab,
                midi_lab_lengths,
                duration_lab,
                duration_lab_lengths,
            ) = expand_to_frame(
                duration_phn, duration_phn_lengths, label, midi, duration_phn
            )

            # for data-parallel
            label_lab = label_lab[:, : label_lab_lengths.max()]
            midi_lab = midi_lab[:, : midi_lab_lengths.max()]
            duration_lab = duration_lab[:, : duration_lab_lengths.max()]

            (
                label_score,
                label_score_lengths,
                midi_score,
                midi_score_lengths,
                duration_score,
                duration_score_phn_lengths,
            ) = expand_to_frame(
                duration_ruled_phn,
                duration_ruled_phn_lengths,
                label,
                midi,
                duration_ruled_phn,
            )

            # for data-parallel
            label_score = label_score[:, : label_score_lengths.max()]
            midi_score = midi_score[:, : midi_score_lengths.max()]
            duration_score = duration_score[:, : duration_score_phn_lengths.max()]
            duration_score_syb = None

        elif isinstance(self.score_feats_extract, SyllableScoreFeats):
            # Remove unused paddings at end
            label_lab = label[:, : label_lengths.max()]
            midi_lab = midi[:, : midi_lengths.max()]
            duration_lab = duration_phn[:, : duration_phn_lengths.max()]

            label_score = label[:, : label_lengths.max()]
            midi_score = midi[:, : midi_lengths.max()]
            duration_score = duration_ruled_phn[:, : duration_ruled_phn_lengths.max()]
            duration_score_syb = duration_syb[:, : duration_syb_lengths.max()]
            slur = slur[:, : slur_lengths.max()]

        input_dict = dict(text=text)
        if decode_config["use_teacher_forcing"] or getattr(self.svs, "use_gst", False):
            if singing is None:
                raise RuntimeError("missing required argument: 'singing'")
            if self.feats_extract is not None:
                feats = self.feats_extract(singing[None])[0][0]
            else:
                # Use precalculated feats (feats_type != raw case)
                feats = singing
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            input_dict.update(feats=feats)
            # if self.svs.require_raw_singing:
            #     input_dict.update(singing=singing)

        if decode_config["use_teacher_forcing"]:
            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    singing[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                input_dict.update(pitch=pitch)

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    singing[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                )[0][0]
            if self.energy_normalize is not None:
                energy = self.energy_normalize(energy[None])[0][0]
            if energy is not None:
                input_dict.update(energy=energy)

        # label
        label = dict()
        if label_lab is not None:
            label_lab = label_lab.to(dtype=torch.long)
            label.update(lab=label_lab)
        if label_score is not None:
            label_score = label_score.to(dtype=torch.long)
            label.update(score=label_score)
        input_dict.update(label=label)

        # melody
        melody = dict()
        if midi_lab is not None:
            midi_lab = midi_lab.to(dtype=torch.long)
            melody.update(lab=midi_lab)
        if midi_score is not None:
            midi_score = midi_score.to(dtype=torch.long)
            melody.update(score=midi_score)
        input_dict.update(melody=melody)

        # duration
        duration = dict()
        if duration_lab is not None:
            duration_lab = duration_lab.to(dtype=torch.long)
            duration.update(lab=duration_lab)
        if duration_score is not None:
            duration_phn_score = duration_score.to(dtype=torch.long)
            duration.update(score_phn=duration_phn_score)
        if duration_score_syb is not None:
            duration_syb_score = duration_score_syb.to(dtype=torch.long)
            duration.update(score_syb=duration_syb_score)
        input_dict.update(duration=duration)

        if slur is not None:
            input_dict.update(slur=slur)
        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        output_dict = self.svs.inference(**input_dict, **decode_config)

        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)

        return output_dict
