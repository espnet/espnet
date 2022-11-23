# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
# Copyright 2022 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Singing-voice-synthesis ESPnet model."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict, Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.feats_extract.score_feats_extract import (
    FrameScoreFeats,
    SyllableScoreFeats,
)
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.nets_utils import pad_list

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


def cal_ds(l, label, midi, beat, ref_len, ref_label, ref_midi, ref_beat):
    ds = []
    i = 0
    j = 0
    while i < l:
        _label = label[i]
        _midi = midi[i]
        _beat = beat[i]
        same = 1
        i += 1
        while i < l and label[i] == _label and midi[i] == _midi and beat[i] == _beat:
            same += 1
            i += 1
        cnt = 0
        while (
            j < ref_len
            and ref_label[j] == _label
            and ref_midi[j] == _midi
            and ref_beat[j] == _beat
        ):
            cnt += 1
            j += 1
        ave = int(cnt / same)
        for k in range(same - 1):
            ds.append(ave)
            cnt -= ave
        ds.append(cnt)
    assert j == ref_len
    return ds


def cal_ds_syb(ds, phn_cnt):
    ds_syb = []
    pos = 0
    for cnt in phn_cnt:
        d = 0
        for k in range(pos, pos + cnt):
            d += ds[k]
        pos += cnt
        for k in range(cnt):
            ds_syb.append(d)
    return ds_syb


class ESPnetSVSModel(AbsESPnetModel):
    """ESPnet model for singing voice synthesis task."""

    def __init__(
        self,
        text_extract: Optional[AbsFeatsExtract],
        feats_extract: Optional[AbsFeatsExtract],
        score_feats_extract: Optional[AbsFeatsExtract],
        label_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        tempo_extract: Optional[AbsFeatsExtract],
        beat_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        svs: AbsSVS,
    ):
        """Initialize ESPnetSVSModel module."""
        assert check_argument_types()
        super().__init__()
        self.text_extract = text_extract
        self.feats_extract = feats_extract
        self.score_feats_extract = score_feats_extract
        self.label_extract = label_extract
        self.pitch_extract = pitch_extract
        self.tempo_extract = tempo_extract
        self.beat_extract = beat_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.svs = svs

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        label_lab: Optional[torch.Tensor] = None,
        label_lab_lengths: Optional[torch.Tensor] = None,
        label_score: Optional[torch.Tensor] = None,
        label_score_lengths: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_lab_lengths: Optional[torch.Tensor] = None,
        midi_score: Optional[torch.Tensor] = None,
        midi_score_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_lab_lengths: Optional[torch.Tensor] = None,
        tempo_score: Optional[torch.Tensor] = None,
        tempo_score_lengths: Optional[torch.Tensor] = None,
        beat_phn: Optional[torch.Tensor] = None,
        beat_phn_lengths: Optional[torch.Tensor] = None,
        beat_ruled_phn: Optional[torch.Tensor] = None,
        beat_ruled_phn_lengths: Optional[torch.Tensor] = None,
        beat_syb: Optional[torch.Tensor] = None,
        beat_syb_lengths: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_lab_lengths: Optional[torch.Tensor] = None,
        beat_score_phn: Optional[torch.Tensor] = None,
        beat_score_phn_lengths: Optional[torch.Tensor] = None,
        beat_score_syb: Optional[torch.Tensor] = None,
        beat_score_syb_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            label_lab (Optional[Tensor]): Label tensor. - phone id sequence
            label_lab_lengths (Optional[Tensor]): Label length tensor (B,).
            label_score (Optional[Tensor]): Label tensor. - phone id sequence
            label_score_lengths (Optional[Tensor]): Label length tensor (B,).
            midi_lab (Optional[Tensor]): Midi tensor.
            midi_lab_lengths (Optional[Tensor]): Midi length tensor (B,).
            midi_score (Optional[Tensor]): Midi tensor.
            midi_score_lengths (Optional[Tensor]): Midi length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            tempo_lab (Optional[Tensor]): Tempo tensor.
            tempo_lab_lengths (Optional[Tensor]): Tempo length tensor (B,).
            tempo_score (Optional[Tensor]): Tempo tensor.
            tempo_score_lengths (Optional[Tensor]): Tempo length tensor (B,).
            beat_lab (Optional[Tensor]): Beat tensor.
            beat_lab_lengths (Optional[Tensor]): Beat length tensor (B,).
            beat_score_phn (Optional[Tensor]): Beat tensor.
            beat_score_phn_lengths (Optional[Tensor]): Beat length tensor (B,).
            beat_score_syb (Optional[Tensor]): Beat tensor.
            beat_score_syb_lengths (Optional[Tensor]): Beat length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            kwargs: "utt_id" is among the input.

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.
        """
        with autocast(False):
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(
                    singing, singing_lengths
                )  # singing to spec feature (frame level)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = singing, singing_lengths

            # Extract auxiliary features
            # score : 128 midi pitch
            # tempo : bpm
            # duration :
            #   input-> phone-id seqence
            #   output -> frame level(take mode from window) or syllable level
            ds = None
            if isinstance(self.score_feats_extract, FrameScoreFeats):
                (
                    label_lab_after,
                    label_lab_lengths_after,
                    midi_lab_after,
                    midi_lab_lengths_after,
                    tempo_lab_after,
                    tempo_lab_lengths_after,
                    beat_lab_after,
                    beat_lab_lengths_after,
                ) = self.score_feats_extract(
                    label=label_lab.unsqueeze(-1),
                    label_lengths=label_lab_lengths,
                    midi=midi_lab.unsqueeze(-1),
                    midi_lengths=midi_lab_lengths,
                    tempo=tempo_lab.unsqueeze(-1),
                    tempo_lengths=tempo_lab_lengths,
                    beat=beat_lab.unsqueeze(-1),
                    beat_lengths=beat_lab_lengths,
                )
                label_lab_after = label_lab_after[
                    :, : label_lab_lengths_after.max()
                ]  # for data-parallel

                # calculate durations, new text & text_length
                # Syllable Level duration info needs phone
                # NOTE(Shuai) Duplicate adjacent phones appear in text files sometimes
                # e.g. oniku_0000000000000000hato_0002
                # 10.951 11.107 sh
                # 11.107 11.336 i
                # 11.336 11.610 i
                # 11.610 11.657 k
                _text_cal = []
                _text_length_cal = []
                ds = []
                for i, _ in enumerate(label_lab_lengths_after):
                    _phone = label_lab_after[i, : label_lab_lengths_after[i]]

                    _output, counts = torch.unique_consecutive(
                        _phone, return_counts=True
                    )

                    _text_cal.append(_output)
                    _text_length_cal.append(len(_output))
                    ds.append(counts)
                ds = pad_list(ds, pad_value=0).to(text.device)
                text = pad_list(_text_cal, pad_value=0).to(
                    text.device, dtype=torch.long
                )
                text_lengths = torch.tensor(_text_length_cal).to(text.device)

                (
                    label_score_after,
                    label_score_lengths_after,
                    midi_score_after,
                    midi_score_lengths_after,
                    tempo_score_after,
                    tempo_score_lengths_after,
                    beat_score_after,
                    beat_score_lengths_after,
                ) = self.score_feats_extract(
                    label=label_score.unsqueeze(-1),
                    label_lengths=label_score_lengths,
                    midi=midi_score.unsqueeze(-1),
                    midi_lengths=midi_score_lengths,
                    tempo=tempo_score.unsqueeze(-1),
                    tempo_lengths=tempo_score_lengths,
                    beat=beat_score_phn.unsqueeze(-1),
                    beat_lengths=beat_score_phn_lengths,
                )

            elif isinstance(self.score_feats_extract, SyllableScoreFeats):
                extractMethod_frame = FrameScoreFeats(
                    fs=self.score_feats_extract.fs,
                    n_fft=self.score_feats_extract.n_fft,
                    win_length=self.score_feats_extract.win_length,
                    hop_length=self.score_feats_extract.hop_length,
                    window=self.score_feats_extract.window,
                    center=self.score_feats_extract.center,
                )

                (
                    labelFrame_lab,
                    labelFrame_lab_lengths,
                    midiFrame_lab,
                    midiFrame_lab_lengths,
                    tempoFrame_lab,
                    tempoFrame_lab_lengths,
                    beatFrame_lab,
                    beatFrame_lab_lengths,
                ) = extractMethod_frame(
                    label=label_lab.unsqueeze(-1),
                    label_lengths=label_lab_lengths,
                    midi=midi_lab.unsqueeze(-1),
                    midi_lengths=midi_lab_lengths,
                    tempo=tempo_lab.unsqueeze(-1),
                    tempo_lengths=tempo_lab_lengths,
                    beat=beat_lab.unsqueeze(-1),
                    beat_lengths=beat_lab_lengths,
                )

                labelFrame_lab = labelFrame_lab[
                    :, : labelFrame_lab_lengths.max()
                ]  # for data-parallel
                midiFrame_lab = midiFrame_lab[
                    :, : midiFrame_lab_lengths.max()
                ]  # for data-parallel

                # calculate durations for feature mapping
                # Syllable Level duration info needs phone & midi
                ds = []
                ds_syb = []

                for i, _ in enumerate(labelFrame_lab_lengths):
                    assert labelFrame_lab_lengths[i] == midiFrame_lab_lengths[i]
                    assert label_lengths[i] == midi_lengths[i]

                    frame_length = labelFrame_lab_lengths[i]
                    _phoneFrame = labelFrame_lab[i, :frame_length]
                    _midiFrame = midiFrame_lab[i, :frame_length]
                    _beatFrame = beatFrame_lab[i, :frame_length]

                    # Clean _phoneFrame & _midiFrame
                    for index in range(frame_length):
                        if _phoneFrame[index] == 0 and _midiFrame[index] == 0:
                            frame_length -= 1
                            feats_lengths[i] -= 1
                    assert frame_length == feats_lengths[i]

                    syllable_length = label_lengths[i]
                    _phoneSyllable = label[i, :syllable_length]
                    _midiSyllable = midi[i, :syllable_length]
                    _beatSyllable = beat_phn[i, :syllable_length]

                    ds_tmp = cal_ds(
                        syllable_length,
                        _phoneSyllable,
                        _midiSyllable,
                        _beatSyllable,
                        frame_length,
                        _phoneFrame,
                        _midiFrame,
                        _beatFrame,
                    )
                    assert sum(ds_tmp) == frame_length
                    ds.append(torch.tensor(ds_tmp))
                    ds_syb_tmp = cal_ds_syb(ds_tmp, phn_cnt[i])
                    ds_syb.append(torch.tensor(ds_syb_tmp))
                ds = pad_list(ds, pad_value=0).to(label_lab.device)
                ds_syb = pad_list(ds_syb, pad_value=0).to(label_lab.device)

                label_lab_lengths_after = label_lengths
                midi_lab_lengths_after = midi_lengths
                tempo_lab_lengths_after = label_lengths
                beat_lab_lengths_after = beat_phn_lengths

                label_score_lengths_after = label_lengths
                midi_score_lengths_after = midi_lengths
                tempo_score_lengths_after = label_lengths
                beat_score_lengths_after = beat_ruled_phn_lengths
                beat_score_syb_lengths_after = beat_syb_lengths

                label_lab_after = label[:, : label_lab_lengths_after.max()]
                midi_lab_after = midi[:, : midi_lab_lengths_after.max()]
                beat_lab_after = beat_phn[:, : beat_lab_lengths_after.max()]
                tempo_lab_after = tempo_lab[:, : tempo_lab_lengths_after.max()]

                label_score_after = label[:, : label_score_lengths_after.max()]
                midi_score_after = midi[:, : midi_score_lengths_after.max()]
                tempo_score_after = tempo_score[:, : tempo_score_lengths_after.max()]
                beat_score_after = beat_ruled_phn[:, : beat_score_lengths_after.max()]
                beat_score_syb_after = beat_syb[:, : beat_score_syb_lengths_after.max()]

            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    input=singing,
                    input_lengths=singing_lengths,
                    feats_lengths=feats_lengths,
                    durations=label_lab,
                    durations_lengths=label_lab_lengths,
                )

            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    singing,
                    singing_lengths,
                    feats_lengths=feats_lengths,
                    durations=label_lab,
                    durations_lengths=label_lab_lengths,
                )

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
            feats=feats,
            feats_lengths=feats_lengths,
            flag_IsValid=flag_IsValid,
        )

        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if midi_lab_after is not None:
            midi_lab = midi_lab_after.to(dtype=torch.long)
            batch.update(midi_lab=midi_lab, midi_lab_lengths=midi_lab_lengths_after)
        if midi_score_after is not None:
            midi_score = midi_score_after.to(dtype=torch.long)
            batch.update(
                midi_score=midi_score, midi_score_lengths=midi_score_lengths_after
            )
        if label_lab_after is not None:
            label_lab = label_lab_after.to(dtype=torch.long)
            batch.update(label_lab=label_lab, label_lab_lengths=label_lab_lengths_after)
        if label_score_after is not None:
            label_score = label_score_after.to(dtype=torch.long)
            batch.update(
                label_score=label_score, label_score_lengths=label_score_lengths_after
            )
        if tempo_lab_after is not None:
            tempo_lab = tempo_lab_after.to(dtype=torch.long)
            batch.update(tempo_lab=tempo_lab, tempo_lab_lengths=tempo_lab_lengths_after)
        if tempo_score_after is not None:
            tempo_score = tempo_score_after.to(dtype=torch.long)
            batch.update(
                tempo_score=tempo_score, tempo_score_lengths=tempo_score_lengths_after
            )
        if beat_lab_after is not None:
            beat_lab = beat_lab_after.to(dtype=torch.long)
            batch.update(beat_lab=beat_lab, beat_lab_lengths=beat_lab_lengths_after)
        if beat_score_after is not None:
            beat_phn_score = beat_score_after.to(dtype=torch.long)
            batch.update(
                beat_phn_score=beat_phn_score,
                beat_phn_score_lengths=beat_score_lengths_after,
            )
        if beat_score_syb_after is not None:
            beat_syb_score = beat_score_syb_after.to(dtype=torch.long)
            batch.update(
                beat_syb_score=beat_syb_score,
                beat_syb_score_lengths=beat_score_syb_lengths_after,
            )
        if ds is not None:
            batch.update(ds=ds)
        if ds_syb is not None:
            batch.update(ds_syb=ds_syb)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if self.svs.require_raw_singing:
            batch.update(singing=singing, singing_lengths=singing_lengths)
        return self.svs(**batch)

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        label_lab: Optional[torch.Tensor] = None,
        label_lab_lengths: Optional[torch.Tensor] = None,
        label_score: Optional[torch.Tensor] = None,
        label_score_lengths: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lengths: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_lab_lengths: Optional[torch.Tensor] = None,
        midi_score: Optional[torch.Tensor] = None,
        midi_score_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_lab_lengths: Optional[torch.Tensor] = None,
        beat_phn: Optional[torch.Tensor] = None,
        beat_phn_lengths: Optional[torch.Tensor] = None,
        beat_ruled_phn: Optional[torch.Tensor] = None,
        beat_ruled_phn_lengths: Optional[torch.Tensor] = None,
        beat_syb: Optional[torch.Tensor] = None,
        beat_syb_lengths: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_lab_lengths: Optional[torch.Tensor] = None,
        tempo_score: Optional[torch.Tensor] = None,
        tempo_score_lengths: Optional[torch.Tensor] = None,
        beat_score_phn: Optional[torch.Tensor] = None,
        beat_score_phn_lengths: Optional[torch.Tensor] = None,
        beat_score_syb: Optional[torch.Tensor] = None,
        beat_score_syb_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            label_lab (Optional[Tensor]): Label tensor. - phone id sequence
            label_lab_lengths (Optional[Tensor]): Label length tensor (B,).
            label_score (Optional[Tensor]): Label tensor. - phone id sequence
            label_score_lengths (Optional[Tensor]): Label length tensor (B,).
            midi_lab (Optional[Tensor]): Midi tensor.
            midi_lab_lengths (Optional[Tensor]): Midi length tensor (B,).
            midi_score (Optional[Tensor]): Midi tensor.
            midi_score_lengths (Optional[Tensor]): Midi length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            tempo_lab (Optional[Tensor]): Tempo tensor.
            tempo_lab_lengths (Optional[Tensor]): Tempo length tensor (B,).
            tempo_score (Optional[Tensor]): Tempo tensor.
            tempo_score_lengths (Optional[Tensor]): Tempo length tensor (B,).
            beat_lab (Optional[Tensor]): Beat tensor.
            beat_lab_lengths (Optional[Tensor]): Beat length tensor (B,).
            beat_score_phn (Optional[Tensor]): Beat tensor.
            beat_score_phn_lengths (Optional[Tensor]): Beat length tensor (B,).
            beat_score_syb (Optional[Tensor]): Beat tensor.
            beat_score_syb_lengths (Optional[Tensor]): Beat length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).

        Returns:
            Dict[str, Tensor]: Dict of features.
        """
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(singing, singing_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            feats, feats_lengths = singing, singing_lengths

        if self.score_feats_extract is not None:
            (
                label_lab_after,
                label_lab_lengths_after,
                midi_lab_after,
                midi_lab_lengths_after,
                tempo_lab_after,
                tempo_lab_lengths_after,
                beat_lab_after,
                beat_lab_lengths_after,
            ) = self.score_feats_extract(
                label=label_lab.unsqueeze(-1),
                label_lengths=label_lab_lengths,
                midi=midi_lab.unsqueeze(-1),
                midi_lengths=midi_lab_lengths,
                tempo=tempo_lab.unsqueeze(-1),
                tempo_lengths=tempo_lab_lengths,
                beat=beat_lab.unsqueeze(-1),
                beat_lengths=beat_lab_lengths,
            )
            (
                label_score_after,
                label_score_lengths_after,
                midi_score_after,
                midi_score_lengths_after,
                tempo_score_after,
                tempo_score_lengths_after,
                beat_score_after,
                beat_score_lengths_after,
            ) = self.score_feats_extract(
                label=label_score.unsqueeze(-1),
                label_lengths=label_score_lengths,
                midi=midi_score.unsqueeze(-1),
                midi_lengths=midi_score_lengths,
                tempo=tempo_score.unsqueeze(-1),
                tempo_lengths=tempo_score_lengths,
                beat=beat_score_phn.unsqueeze(-1),
                beat_lengths=beat_score_phn_lengths,
            )

        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                input=singing,
                input_lengths=singing_lengths,
                feats_lengths=feats_lengths,
                durations=label_lab,
                durations_lengths=label_lab_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                singing,
                singing_lengths,
                feats_lengths=feats_lengths,
                durations=label_lab,
                durations_lengths=label_lab_lengths,
            )

        # store in dict
        feats_dict = dict(feats=feats, feats_lengths=feats_lengths)
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict

    def inference(
        self,
        text: torch.Tensor,
        singing: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label_lab: Optional[torch.Tensor] = None,
        label_score: Optional[torch.Tensor] = None,
        phn_cnt: Optional[torch.Tensor] = None,
        midi: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_score: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_score: Optional[torch.Tensor] = None,
        beat_phn: Optional[torch.Tensor] = None,
        beat_ruled_phn: Optional[torch.Tensor] = None,
        beat_syb: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_score_phn: Optional[torch.Tensor] = None,
        beat_score_syb: Optional[torch.Tensor] = None,
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
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            label (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            tempo (Optional[Tensor): Tempo tensor.
            beat (Optional[Tensor): Beat tensor.
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.
        """
        label_lengths = torch.tensor([len(label)])
        midi_lengths = torch.tensor([len(midi)])
        beat_phn_lengths = torch.tensor([len(beat_phn)])
        beat_ruled_phn_lengths = torch.tensor([len(beat_ruled_phn)])
        beat_syb_lengths = torch.tensor([len(beat_syb)])

        label_lab_lengths = torch.tensor([len(label_lab)])
        midi_lab_lengths = torch.tensor([len(midi_lab)])
        tempo_lab_lengths = torch.tensor([len(tempo_lab)])
        beat_lab_lengths = torch.tensor([len(beat_lab)])
        assert (
            label_lab_lengths == midi_lab_lengths
            and label_lab_lengths == tempo_lab_lengths
            and tempo_lab_lengths == beat_lab_lengths
        )

        label_score_lengths = torch.tensor([len(label_score)])
        midi_score_lengths = torch.tensor([len(midi_score)])
        tempo_score_lengths = torch.tensor([len(tempo_score)])
        beat_score_lengths = torch.tensor([len(beat_score_phn)])
        assert (
            label_score_lengths == midi_score_lengths
            and label_score_lengths == tempo_score_lengths
            and tempo_score_lengths == beat_score_lengths
        )

        # unsqueeze of singing needed otherwise causing error in STFT dimension
        # for data-parallel
        text = text.unsqueeze(0)

        label = label.unsqueeze(0)
        midi = midi.unsqueeze(0)
        beat_phn = beat_phn.unsqueeze(0)
        beat_ruled_phn = beat_ruled_phn.unsqueeze(0)
        beat_syb = beat_syb.unsqueeze(0)
        phn_cnt = phn_cnt.unsqueeze(0)

        label_lab = label_lab.unsqueeze(0)
        midi_lab = midi_lab.unsqueeze(0)
        tempo_lab = tempo_lab.unsqueeze(0)
        beat_lab = beat_lab.unsqueeze(0)

        label_score = label_score.unsqueeze(0)
        midi_score = midi_score.unsqueeze(0)
        tempo_score = tempo_score.unsqueeze(0)
        beat_score_phn = beat_score_phn.unsqueeze(0)
        beat_score_syb = beat_score_syb.unsqueeze(0)

        # Extract auxiliary features
        # score : 128 midi pitch
        # tempo : bpm
        # duration :
        #   input-> phone-id seqence
        #   output -> frame level or syllable level
        ds = None
        batch_size = text.size(0)
        assert batch_size == 1
        if isinstance(self.score_feats_extract, FrameScoreFeats):
            (
                label_lab_after,
                label_lab_lengths_after,
                midi_lab_after,
                midi_lab_lengths_after,
                tempo_lab_after,
                tempo_lab_lengths_after,
                beat_lab_after,
                beat_lab_lengths_after,
            ) = self.score_feats_extract(
                label=label_lab.unsqueeze(-1),
                label_lengths=label_lab_lengths,
                midi=midi_lab.unsqueeze(-1),
                midi_lengths=midi_lab_lengths,
                tempo=tempo_lab.unsqueeze(-1),
                tempo_lengths=tempo_lab_lengths,
                beat=beat_lab.unsqueeze(-1),
                beat_lengths=beat_lab_lengths,
            )

            # calculate durations, new text & text_length
            # Syllable Level duration info needs phone
            # NOTE(Shuai) Duplicate adjacent phones will appear in text files sometimes
            # e.g. oniku_0000000000000000hato_0002
            # 10.951 11.107 sh
            # 11.107 11.336 i
            # 11.336 11.610 i
            # 11.610 11.657 k
            _text_cal = []
            _text_length_cal = []
            ds = []
            for i in range(batch_size):
                _phone = label_lab_after[i]

                _output, counts = torch.unique_consecutive(_phone, return_counts=True)

                _text_cal.append(_output)
                _text_length_cal.append(len(_output))
                ds.append(counts)
            ds = pad_list(ds, pad_value=0).to(text.device)
            text = pad_list(_text_cal, pad_value=0).to(text.device, dtype=torch.long)

            (
                label_score_after,
                label_score_lengths_after,
                midi_score_after,
                midi_score_lengths_after,
                tempo_score_after,
                tempo_score_lengths_after,
                beat_score_after,
                beat_score_lengths_after,
            ) = self.score_feats_extract(
                label=label_score.unsqueeze(-1),
                label_lengths=label_score_lengths,
                midi=midi_score.unsqueeze(-1),
                midi_lengths=midi_score_lengths,
                tempo=tempo_score.unsqueeze(-1),
                tempo_lengths=tempo_score_lengths,
                beat=beat_score_phn.unsqueeze(-1),
                beat_lengths=beat_score_phn_lengths,
            )

        elif isinstance(self.score_feats_extract, SyllableScoreFeats):
            extractMethod_frame = FrameScoreFeats(
                fs=self.score_feats_extract.fs,
                n_fft=self.score_feats_extract.n_fft,
                win_length=self.score_feats_extract.win_length,
                hop_length=self.score_feats_extract.hop_length,
                window=self.score_feats_extract.window,
                center=self.score_feats_extract.center,
            )

            (
                labelFrame_lab,
                labelFrame_lab_lengths,
                midiFrame_lab,
                midiFrame_lab_lengths,
                tempoFrame_lab,
                tempoFrame_lab_lengths,
                beatFrame_lab,
                beatFrame_lab_lengths,
            ) = extractMethod_frame(
                label=label_lab.unsqueeze(-1),
                label_lengths=label_lab_lengths,
                midi=midi_lab.unsqueeze(-1),
                midi_lengths=midi_lab_lengths,
                tempo=tempo_lab.unsqueeze(-1),
                tempo_lengths=tempo_lab_lengths,
                beat=beat_lab.unsqueeze(-1),
                beat_lengths=beat_lab_lengths,
            )

            labelFrame_lab = labelFrame_lab[
                :, : labelFrame_lab_lengths.max()
            ]  # for data-parallel
            midiFrame_lab = midiFrame_lab[
                :, : midiFrame_lab_lengths.max()
            ]  # for data-parallel

            # calculate durations, represent syllable encoder outputs to feats mapping
            # Syllable Level duration info needs phone & midi
            ds = []
            ds_syb = []

            for i, _ in enumerate(labelFrame_lab_lengths):
                assert labelFrame_lab_lengths[i] == midiFrame_lab_lengths[i]
                assert label_lab_lengths[i] == midi_lab_lengths[i]

                frame_length = labelFrame_lab_lengths[i]
                _phoneFrame = labelFrame_lab[i, :frame_length]
                _midiFrame = midiFrame_lab[i, :frame_length]
                _beatFrame = beatFrame_lab[i, :frame_length]

                # Clean _phoneFrame & _midiFrame
                for index in range(frame_length):
                    if _phoneFrame[index] == 0 and _midiFrame[index] == 0:
                        frame_length -= 1

                syllable_length = label_lengths[i]
                _phoneSyllable = label[i, :syllable_length]
                _midiSyllable = midi[i, :syllable_length]
                _beatSyllable = beat_phn[i, :syllable_length]

                ds_tmp = cal_ds(
                    syllable_length,
                    _phoneSyllable,
                    _midiSyllable,
                    _beatSyllable,
                    frame_length,
                    _phoneFrame,
                    _midiFrame,
                    _beatFrame,
                )
                assert sum(ds_tmp) == frame_length
                ds.append(torch.tensor(ds_tmp))
                ds_syb_tmp = cal_ds_syb(ds_tmp, phn_cnt[i])
                ds_syb.append(torch.tensor(ds_syb_tmp))
            ds = pad_list(ds, pad_value=0).to(label_lab.device)
            ds_syb = pad_list(ds_syb, pad_value=0).to(label_lab.device)

            label_lab_lengths_after = label_lengths
            midi_lab_lengths_after = midi_lengths
            tempo_lab_lengths_after = label_lengths
            beat_lab_lengths_after = beat_phn_lengths

            label_score_lengths_after = label_lengths
            midi_score_lengths_after = midi_lengths
            tempo_score_lengths_after = label_lengths
            beat_score_lengths_after = beat_ruled_phn_lengths
            beat_score_syb_lengths_after = beat_syb_lengths

            label_lab_after = label[:, : label_lab_lengths_after.max()]
            midi_lab_after = midi[:, : midi_lab_lengths_after.max()]
            beat_lab_after = beat_phn[:, : beat_lab_lengths_after.max()]
            tempo_lab_after = tempo_lab[:, : tempo_lab_lengths_after.max()]

            label_score_after = label[:, : label_score_lengths_after.max()]
            midi_score_after = midi[:, : midi_score_lengths_after.max()]
            tempo_score_after = tempo_score[:, : tempo_score_lengths_after.max()]
            beat_score_after = beat_ruled_phn[:, : beat_score_lengths_after.max()]
            beat_score_syb_after = beat_syb[:, : beat_score_syb_lengths_after.max()]

        input_dict = dict(text=text)

        if midi_lab_after is not None:
            midi_lab = midi_lab_after.to(dtype=torch.long)
            input_dict["midi_lab"] = midi_lab
        if midi_score_after is not None:
            midi_score = midi_score_after.to(dtype=torch.long)
            input_dict["midi_score"] = midi_score
        if label_lab_after is not None:
            label_lab = label_lab_after.to(dtype=torch.long)
            input_dict["label_lab"] = label_lab
        if label_score_after is not None:
            label_score = label_score_after.to(dtype=torch.long)
            input_dict["label_score"] = label_score
        if ds is not None:
            input_dict.update(ds=ds)
        if ds_syb is not None:
            input_dict.update(ds_syb=ds_syb)
        if tempo_lab_after is not None:
            tempo_lab = tempo_lab_after.to(dtype=torch.long)
            input_dict.update(tempo_lab=tempo_lab)
        if tempo_score_after is not None:
            tempo_score = tempo_score_after.to(dtype=torch.long)
            input_dict.update(tempo_score=tempo_score)
        if beat_lab_after is not None:
            beat_lab = beat_lab_after.to(dtype=torch.long)
            input_dict.update(beat_lab=beat_lab)
        if beat_score_after is not None:
            beat_phn_score = beat_score_after.to(dtype=torch.long)
            input_dict.update(beat_phn_score=beat_phn_score)
        if beat_score_syb_after is not None:
            beat_syb_score = beat_score_syb_after.to(dtype=torch.long)
            input_dict.update(beat_syb_score=beat_syb_score)
        if pitch is not None:
            batch.update(pitch=pitch)
        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        outs, probs, att_ws = self.svs.inference(**input_dict)

        if self.normalize is not None:
            # NOTE: normalize.inverse is in-place operation
            outs_denorm = self.normalize.inverse(outs.clone()[None])[0][0]
        else:
            outs_denorm = outs

        return outs, outs_denorm, probs, att_ws
