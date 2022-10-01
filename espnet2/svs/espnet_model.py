# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
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
        label_lab: Optional[torch.Tensor] = None,
        label_lab_lengths: Optional[torch.Tensor] = None,
        label_xml: Optional[torch.Tensor] = None,
        label_xml_lengths: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_lab_lengths: Optional[torch.Tensor] = None,
        midi_xml: Optional[torch.Tensor] = None,
        midi_xml_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_lab_lengths: Optional[torch.Tensor] = None,
        tempo_xml: Optional[torch.Tensor] = None,
        tempo_xml_lengths: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_lab_lengths: Optional[torch.Tensor] = None,
        beat_xml: Optional[torch.Tensor] = None,
        beat_xml_lengths: Optional[torch.Tensor] = None,
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
            label_xml (Optional[Tensor]): Label tensor. - phone id sequence
            label_xml_lengths (Optional[Tensor]): Label length tensor (B,).
            midi_lab (Optional[Tensor]): Midi tensor.
            midi_lab_lengths (Optional[Tensor]): Midi length tensor (B,).
            midi_xml (Optional[Tensor]): Midi tensor.
            midi_xml_lengths (Optional[Tensor]): Midi length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            tempo_lab (Optional[Tensor]): Tempo tensor.
            tempo_lab_lengths (Optional[Tensor]): Tempo length tensor (B,).
            tempo_xml (Optional[Tensor]): Tempo tensor.
            tempo_xml_lengths (Optional[Tensor]): Tempo length tensor (B,).
            beat_lab (Optional[Tensor]): Beat tensor.
            beat_lab_lengths (Optional[Tensor]): Beat length tensor (B,).
            beat_xml (Optional[Tensor]): Beat tensor.
            beat_xml_lengths (Optional[Tensor]): Beat length tensor (B,).
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
            #   input-> phone-id seqence | output -> frame level(take mode from window) or syllable level
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
                # NOTE(Shuai) Duplicate adjacent phones will appear in text files sometimes
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
                    label_xml_after,
                    label_xml_lengths_after,
                    midi_xml_after,
                    midi_xml_lengths_after,
                    tempo_xml_after,
                    tempo_xml_lengths_after,
                    beat_xml_after,
                    beat_xml_lengths_after,
                ) = self.score_feats_extract(
                    label=label_xml.unsqueeze(-1),
                    label_lengths=label_xml_lengths,
                    midi=midi_xml.unsqueeze(-1),
                    midi_lengths=midi_xml_lengths,
                    tempo=tempo_xml.unsqueeze(-1),
                    tempo_lengths=tempo_xml_lengths,
                    beat=beat_xml.unsqueeze(-1),
                    beat_lengths=beat_xml_lengths,
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

                # Extract Syllable Level label, midi, tempo, beat information from Frame Level
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
                    label=labelFrame_lab,
                    label_lengths=labelFrame_lab_lengths,
                    midi=midiFrame_lab,
                    midi_lengths=midiFrame_lab_lengths,
                    tempo=tempoFrame_lab,
                    tempo_lengths=tempoFrame_lab_lengths,
                    beat=beatFrame_lab,
                    beat_lengths=beatFrame_lab_lengths,
                )

                (
                    labelFrame_xml,
                    labelFrame_xml_lengths,
                    midiFrame_xml,
                    midiFrame_xml_lengths,
                    tempoFrame_xml,
                    tempoFrame_xml_lengths,
                    beatFrame_xml,
                    beatFrame_xml_lengths,
                ) = extractMethod_frame(
                    label=label_xml.unsqueeze(-1),
                    label_lengths=label_xml_lengths,
                    midi=midi_xml.unsqueeze(-1),
                    midi_lengths=midi_xml_lengths,
                    tempo=tempo_xml.unsqueeze(-1),
                    tempo_lengths=tempo_xml_lengths,
                    beat=beat_xml.unsqueeze(-1),
                    beat_lengths=beat_xml_lengths,
                )

                labelFrame_xml = labelFrame_xml[
                    :, : labelFrame_xml_lengths.max()
                ]  # for data-parallel
                midiFrame_xml = midiFrame_xml[
                    :, : midiFrame_xml_lengths.max()
                ]  # for data-parallel

                # Extract Syllable Level label, midi, tempo, beat information from Frame Level
                (
                    label_xml_after,
                    label_xml_lengths_after,
                    midi_xml_after,
                    midi_xml_lengths_after,
                    tempo_xml_after,
                    tempo_xml_lengths_after,
                    beat_xml_after,
                    beat_xml_lengths_after,
                ) = self.score_feats_extract(
                    label=labelFrame_xml,
                    label_lengths=labelFrame_xml_lengths,
                    midi=midiFrame_xml,
                    midi_lengths=midiFrame_xml_lengths,
                    tempo=tempoFrame_xml,
                    tempo_lengths=tempoFrame_xml_lengths,
                    beat=beatFrame_xml,
                    beat_lengths=beatFrame_xml_lengths,
                )

                # calculate durations, represent syllable encoder outputs to feats mapping
                # Syllable Level duration info needs phone & midi
                ds = []
                for i, _ in enumerate(labelFrame_lab_lengths):
                    assert labelFrame_lab_lengths[i] == midiFrame_lab_lengths[i]
                    assert label_lab_lengths[i] == midi_lab_lengths[i]

                    frame_length = labelFrame_lab_lengths[i]
                    _phoneFrame = labelFrame_lab[i, :frame_length]
                    _midiFrame = midiFrame_lab[i, :frame_length]

                    # Clean _phoneFrame & _midiFrame
                    for index in range(frame_length):
                        if _phoneFrame[index] == 0 and _midiFrame[index] == 0:
                            frame_length -= 1
                            feats_lengths[i] -= 1

                    syllable_length = label_lab_lengths_after[i]
                    _phoneSyllable = label_lab_after[i, :syllable_length]
                    _midiSyllable = midi_lab_after[i, :syllable_length]

                    start_index = 0
                    ds_tmp = []
                    flag_finish = 0
                    for index in range(syllable_length):
                        _findPhone = _phoneSyllable[index]
                        _findMidi = _midiSyllable[index]
                        _length = 0
                        if flag_finish == 1:
                            # Remove unused note(label = 0, midi = 0)
                            for xml_index in range(index, label_xml_lengths_after[i]):
                                assert label_xml_after[i][xml_index] == 0
                                assert midi_xml_after[i][xml_index] == 0
                                tempo_xml_after[i][xml_index] = 0
                                beat_xml_after[i][xml_index] = 0
                            # Fix error in _phoneSyllable & _midiSyllable
                            assert label_lab_after[i, index] == 0
                            assert midi_lab_after[i, index] == 0
                            tempo_lab_after[i, index] = 0
                            beat_lab_after[i, index] = 0
                            label_lab_lengths_after[i] -= 1
                            midi_lab_lengths_after[i] -= 1
                            tempo_lab_lengths_after[i] -= 1
                            beat_lab_lengths_after[i] -= 1
                            label_xml_lengths_after[i] = label_lab_lengths_after[i]
                            midi_xml_lengths_after[i] = midi_lab_lengths_after[i]
                            tempo_xml_lengths_after[i] = tempo_lab_lengths_after[i]
                            beat_xml_lengths_after[i] = beat_lab_lengths_after[i]
                        else:
                            if (
                                _findPhone != label_xml_after[i][index]
                                or _findMidi != midi_xml_after[i][index]
                            ):
                                for xml_start in range(
                                    index + 1, label_xml_lengths_after[i] - 1
                                ):
                                    if (
                                        _findPhone == label_xml_after[i][xml_start]
                                        and _findMidi == midi_xml_after[i][xml_start]
                                    ):
                                        delta = xml_start - index
                                        for xml_index in range(
                                            index, label_xml_lengths_after[i] - delta
                                        ):
                                            label_xml_after[i][
                                                xml_index
                                            ] = label_xml_after[i][xml_index + delta]
                                            midi_xml_after[i][
                                                xml_index
                                            ] = midi_xml_after[i][xml_index + delta]
                                            tempo_xml_after[i][
                                                xml_index
                                            ] = tempo_xml_after[i][xml_index + delta]
                                            beat_xml_after[i][
                                                xml_index
                                            ] = beat_xml_after[i][xml_index + delta]
                                        for xml_index in range(
                                            label_xml_lengths_after[i] - delta,
                                            label_xml_lengths_after[i],
                                        ):
                                            label_xml_after[i][xml_index] = 0
                                            midi_xml_after[i][xml_index] = 0
                                            tempo_xml_after[i][xml_index] = 0
                                            beat_xml_after[i][xml_index] = 0
                                        break
                                assert _findPhone == label_xml_after[i][index]
                                assert _findMidi == midi_xml_after[i][index]
                            for indexFrame in range(start_index, frame_length):
                                if (
                                    _phoneFrame[indexFrame] == _findPhone
                                    and _midiFrame[indexFrame] == _findMidi
                                ):
                                    _length += 1
                                else:
                                    ds_tmp.append(_length)
                                    start_index = indexFrame
                                    break
                                if indexFrame == frame_length - 1:
                                    flag_finish = 1
                                    ds_tmp.append(_length)
                                    start_index = indexFrame
                                    if syllable_length < label_xml_lengths_after[i]:
                                        for xml_index in range(
                                            syllable_length, label_xml_lengths_after[i]
                                        ):
                                            assert label_xml_after[i][xml_index] == 0
                                            assert midi_xml_after[i][xml_index] == 0
                                            tempo_xml_after[i][xml_index] == 0
                                            beat_xml_after[i][xml_index] == 0
                                        label_xml_lengths_after[
                                            i
                                        ] = label_lab_lengths_after[i]
                                        midi_xml_lengths_after[
                                            i
                                        ] = midi_lab_lengths_after[i]
                                        tempo_xml_lengths_after[
                                            i
                                        ] = tempo_lab_lengths_after[i]
                                        beat_xml_lengths_after[
                                            i
                                        ] = beat_lab_lengths_after[i]
                                    break
                    assert (
                        sum(ds_tmp) == frame_length and sum(ds_tmp) == feats_lengths[i]
                    )

                    ds.append(torch.tensor(ds_tmp))
                ds = pad_list(ds, pad_value=0).to(label_lab_after.device)

            #
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
        if midi_lab_after is not None and pitch is None:
            midi_lab = midi_lab_after.to(dtype=torch.long)
            batch.update(midi_lab=midi_lab, midi_lab_lengths=midi_lab_lengths_after)
        if midi_xml_after is not None and pitch is None:
            midi_xml = midi_xml_after.to(dtype=torch.long)
            batch.update(midi_xml=midi_xml, midi_xml_lengths=midi_xml_lengths_after)
        if label_lab_after is not None:
            label_lab = label_lab_after.to(dtype=torch.long)
            batch.update(label_lab=label_lab, label_lab_lengths=label_lab_lengths_after)
        if label_xml_after is not None:
            label_xml = label_xml_after.to(dtype=torch.long)
            batch.update(label_xml=label_xml, label_xml_lengths=label_xml_lengths_after)
        if tempo_lab_after is not None:
            tempo_lab = tempo_lab_after.to(dtype=torch.long)
            batch.update(tempo_lab=tempo_lab, tempo_lab_lengths=tempo_lab_lengths_after)
        if tempo_xml_after is not None:
            tempo_xml = tempo_xml_after.to(dtype=torch.long)
            batch.update(tempo_xml=tempo_xml, tempo_xml_lengths=tempo_xml_lengths_after)
        if beat_lab_after is not None:
            beat_lab = beat_lab_after.to(dtype=torch.long)
            batch.update(beat_lab=beat_lab, beat_lab_lengths=beat_lab_lengths_after)
        if beat_xml_after is not None:
            beat_xml = beat_xml_after.to(dtype=torch.long)
            batch.update(beat_xml=beat_xml, beat_xml_lengths=beat_xml_lengths_after)
        if ds is not None:
            batch.update(ds=ds)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(midi=pitch, midi_lengths=pitch_lengths)
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
        label_lab: Optional[torch.Tensor] = None,
        label_lab_lengths: Optional[torch.Tensor] = None,
        label_xml: Optional[torch.Tensor] = None,
        label_xml_lengths: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_lab_lengths: Optional[torch.Tensor] = None,
        midi_xml: Optional[torch.Tensor] = None,
        midi_xml_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_lab_lengths: Optional[torch.Tensor] = None,
        tempo_xml: Optional[torch.Tensor] = None,
        tempo_xml_lengths: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_lab_lengths: Optional[torch.Tensor] = None,
        beat_xml: Optional[torch.Tensor] = None,
        beat_xml_lengths: Optional[torch.Tensor] = None,
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
            label_xml (Optional[Tensor]): Label tensor. - phone id sequence
            label_xml_lengths (Optional[Tensor]): Label length tensor (B,).
            midi_lab (Optional[Tensor]): Midi tensor.
            midi_lab_lengths (Optional[Tensor]): Midi length tensor (B,).
            midi_xml (Optional[Tensor]): Midi tensor.
            midi_xml_lengths (Optional[Tensor]): Midi length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            tempo_lab (Optional[Tensor]): Tempo tensor.
            tempo_lab_lengths (Optional[Tensor]): Tempo length tensor (B,).
            tempo_xml (Optional[Tensor]): Tempo tensor.
            tempo_xml_lengths (Optional[Tensor]): Tempo length tensor (B,).
            beat_lab (Optional[Tensor]): Beat tensor.
            beat_lab_lengths (Optional[Tensor]): Beat length tensor (B,).
            beat_xml (Optional[Tensor]): Beat tensor.
            beat_xml_lengths (Optional[Tensor]): Beat length tensor (B,).
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
                label_xml_after,
                label_xml_lengths_after,
                midi_xml_after,
                midi_xml_lengths_after,
                tempo_xml_after,
                tempo_xml_lengths_after,
                beat_xml_after,
                beat_xml_lengths_after,
            ) = self.score_feats_extract(
                label=label_xml.unsqueeze(-1),
                label_lengths=label_xml_lengths,
                midi=midi_xml.unsqueeze(-1),
                midi_lengths=midi_xml_lengths,
                tempo=tempo_xml.unsqueeze(-1),
                tempo_lengths=tempo_xml_lengths,
                beat=beat_xml.unsqueeze(-1),
                beat_lengths=beat_xml_lengths,
            )

        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                input=pitch.unsqueeze(-1),
                input_lengths=pitch_lengths,
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
        label_lab: Optional[torch.Tensor] = None,
        label_xml: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_xml: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_xml: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_xml: Optional[torch.Tensor] = None,
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
        label_lab_lengths = torch.tensor([len(label_lab)])
        midi_lab_lengths = torch.tensor([len(midi_lab)])
        tempo_lab_lengths = torch.tensor([len(tempo_lab)])
        beat_lab_lengths = torch.tensor([len(beat_lab)])
        assert (
            label_lab_lengths == midi_lab_lengths
            and label_lab_lengths == tempo_lab_lengths
            and tempo_lab_lengths == beat_lab_lengths
        )

        label_xml_lengths = torch.tensor([len(label_xml)])
        midi_xml_lengths = torch.tensor([len(midi_xml)])
        tempo_xml_lengths = torch.tensor([len(tempo_xml)])
        beat_xml_lengths = torch.tensor([len(beat_xml)])
        assert (
            label_xml_lengths == midi_xml_lengths
            and label_xml_lengths == tempo_xml_lengths
            and tempo_xml_lengths == beat_xml_lengths
        )

        # unsqueeze of singing must be here, or it'll cause error in the return dim of STFT
        # for data-parallel
        text = text.unsqueeze(0)
        label_lab = label_lab.unsqueeze(0)
        midi_lab = midi_lab.unsqueeze(0)
        tempo_lab = tempo_lab.unsqueeze(0)
        beat_lab = beat_lab.unsqueeze(0)

        label_xml = label_xml.unsqueeze(0)
        midi_xml = midi_xml.unsqueeze(0)
        tempo_xml = tempo_xml.unsqueeze(0)
        beat_xml = beat_xml.unsqueeze(0)

        # Extract auxiliary features
        # score : 128 midi pitch
        # tempo : bpm
        # duration :
        #   input-> phone-id seqence | output -> frame level(取众数 from window) or syllable level
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
            text_lengths = torch.tensor(_text_length_cal).to(text.device)

            (
                label_xml_after,
                label_xml_lengths_after,
                midi_xml_after,
                midi_xml_lengths_after,
                tempo_xml_after,
                tempo_xml_lengths_after,
                beat_xml_after,
                beat_xml_lengths_after,
            ) = self.score_feats_extract(
                label=label_xml.unsqueeze(-1),
                label_lengths=label_xml_lengths,
                midi=midi_xml.unsqueeze(-1),
                midi_lengths=midi_xml_lengths,
                tempo=tempo_xml.unsqueeze(-1),
                tempo_lengths=tempo_xml_lengths,
                beat=beat_xml.unsqueeze(-1),
                beat_lengths=beat_xml_lengths,
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

            # Extract Syllable Level label, midi, tempo, beat information from Frame Level
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
                label=labelFrame_lab,
                label_lengths=labelFrame_lab_lengths,
                midi=midiFrame_lab,
                midi_lengths=midiFrame_lab_lengths,
                tempo=tempoFrame_lab,
                tempo_lengths=tempoFrame_lab_lengths,
                beat=beatFrame_lab,
                beat_lengths=beatFrame_lab_lengths,
            )

            (
                labelFrame_xml,
                labelFrame_xml_lengths,
                midiFrame_xml,
                midiFrame_xml_lengths,
                tempoFrame_xml,
                tempoFrame_xml_lengths,
                beatFrame_xml,
                beatFrame_xml_lengths,
            ) = extractMethod_frame(
                label=label_xml.unsqueeze(-1),
                label_lengths=label_xml_lengths,
                midi=midi_xml.unsqueeze(-1),
                midi_lengths=midi_xml_lengths,
                tempo=tempo_xml.unsqueeze(-1),
                tempo_lengths=tempo_xml_lengths,
                beat=beat_xml.unsqueeze(-1),
                beat_lengths=beat_xml_lengths,
            )

            labelFrame_xml = labelFrame_xml[
                :, : labelFrame_xml_lengths.max()
            ]  # for data-parallel
            midiFrame_xml = midiFrame_xml[
                :, : midiFrame_xml_lengths.max()
            ]  # for data-parallel

            # Extract Syllable Level label, midi, tempo, beat information from Frame Level
            (
                label_xml_after,
                label_xml_lengths_after,
                midi_xml_after,
                midi_xml_lengths_after,
                tempo_xml_after,
                tempo_xml_lengths_after,
                beat_xml_after,
                beat_xml_lengths_after,
            ) = self.score_feats_extract(
                label=labelFrame_xml,
                label_lengths=labelFrame_xml_lengths,
                midi=midiFrame_xml,
                midi_lengths=midiFrame_xml_lengths,
                tempo=tempoFrame_xml,
                tempo_lengths=tempoFrame_xml_lengths,
                beat=beatFrame_xml,
                beat_lengths=beatFrame_xml_lengths,
            )

            # calculate durations, represent syllable encoder outputs to feats mapping
            # Syllable Level duration info needs phone & midi
            ds = []
            for i, _ in enumerate(labelFrame_lab_lengths):
                assert labelFrame_lab_lengths[i] == midiFrame_lab_lengths[i]
                assert label_lab_lengths[i] == midi_lab_lengths[i]

                frame_length = labelFrame_lab_lengths[i]
                _phoneFrame = labelFrame_lab[i, :frame_length]
                _midiFrame = midiFrame_lab[i, :frame_length]

                # Clean _phoneFrame & _midiFrame
                for index in range(frame_length):
                    if _phoneFrame[index] == 0 and _midiFrame[index] == 0:
                        frame_length -= 1

                syllable_length = label_lab_lengths_after[i]
                _phoneSyllable = label_lab_after[i, :syllable_length]
                _midiSyllable = midi_lab_after[i, :syllable_length]

                start_index = 0
                ds_tmp = []
                flag_finish = 0
                for index in range(syllable_length):
                    _findPhone = _phoneSyllable[index]
                    _findMidi = _midiSyllable[index]
                    _length = 0
                    if flag_finish == 1:
                        # Remove unused note(label = 0, midi = 0)
                        for xml_index in range(index, label_xml_lengths_after[i]):
                            assert label_xml_after[i][xml_index] == 0
                            assert midi_xml_after[i][xml_index] == 0
                            tempo_xml_after[i][xml_index] = 0
                            beat_xml_after[i][xml_index] = 0
                        # Fix error in _phoneSyllable & _midiSyllable
                        assert label_lab_after[i, index] == 0
                        assert midi_lab_after[i, index] == 0
                        tempo_lab_after[i, index] = 0
                        beat_lab_after[i, index] = 0
                        label_lab_lengths_after[i] -= 1
                        midi_lab_lengths_after[i] -= 1
                        tempo_lab_lengths_after[i] -= 1
                        beat_lab_lengths_after[i] -= 1
                        label_xml_lengths_after[i] = label_lab_lengths_after[i]
                        midi_xml_lengths_after[i] = midi_lab_lengths_after[i]
                        tempo_xml_lengths_after[i] = tempo_lab_lengths_after[i]
                        beat_xml_lengths_after[i] = beat_lab_lengths_after[i]
                    else:
                        if (
                            _findPhone != label_xml_after[i][index]
                            or _findMidi != midi_xml_after[i][index]
                        ):
                            for xml_start in range(
                                index + 1, label_xml_lengths_after[i] - 1
                            ):
                                if (
                                    _findPhone == label_xml_after[i][xml_start]
                                    and _findMidi == midi_xml_after[i][xml_start]
                                ):
                                    delta = xml_start - index
                                    for xml_index in range(
                                        index, label_xml_lengths_after[i] - delta
                                    ):
                                        label_xml_after[i][xml_index] = label_xml_after[
                                            i
                                        ][xml_index + delta]
                                        midi_xml_after[i][xml_index] = midi_xml_after[
                                            i
                                        ][xml_index + delta]
                                        tempo_xml_after[i][xml_index] = tempo_xml_after[
                                            i
                                        ][xml_index + delta]
                                        beat_xml_after[i][xml_index] = beat_xml_after[
                                            i
                                        ][xml_index + delta]
                                    for xml_index in range(
                                        label_xml_lengths_after[i] - delta,
                                        label_xml_lengths_after[i],
                                    ):
                                        label_xml_after[i][xml_index] = 0
                                        midi_xml_after[i][xml_index] = 0
                                        tempo_xml_after[i][xml_index] = 0
                                        beat_xml_after[i][xml_index] = 0
                                    break
                            assert _findPhone == label_xml_after[i][index]
                            assert _findMidi == midi_xml_after[i][index]
                        for indexFrame in range(start_index, frame_length):
                            if (
                                _phoneFrame[indexFrame] == _findPhone
                                and _midiFrame[indexFrame] == _findMidi
                            ):
                                _length += 1
                            else:
                                ds_tmp.append(_length)
                                start_index = indexFrame
                                break
                            if indexFrame == frame_length - 1:
                                flag_finish = 1
                                ds_tmp.append(_length)
                                start_index = indexFrame
                                if syllable_length < label_xml_lengths_after[i]:
                                    for xml_index in range(
                                        syllable_length, label_xml_lengths_after[i]
                                    ):
                                        assert label_xml_after[i][xml_index] == 0
                                        assert midi_xml_after[i][xml_index] == 0
                                        tempo_xml_after[i][xml_index] == 0
                                        beat_xml_after[i][xml_index] == 0
                                    label_xml_lengths_after[
                                        i
                                    ] = label_lab_lengths_after[i]
                                    midi_xml_lengths_after[i] = midi_lab_lengths_after[
                                        i
                                    ]
                                    tempo_xml_lengths_after[
                                        i
                                    ] = tempo_lab_lengths_after[i]
                                    beat_xml_lengths_after[i] = beat_lab_lengths_after[
                                        i
                                    ]
                                break
                assert sum(ds_tmp) == frame_length

                ds.append(torch.tensor(ds_tmp))
            ds = pad_list(ds, pad_value=0).to(label_lab_after.device)

        input_dict = dict(text=text)

        if midi_lab_after is not None and pitch is None:
            midi_lab = midi_lab_after.to(dtype=torch.long)
            input_dict["midi_lab"] = midi_lab
        if midi_xml_after is not None and pitch is None:
            midi_xml = midi_xml_after.to(dtype=torch.long)
            input_dict["midi_xml"] = midi_xml
        if label_lab_after is not None:
            label_lab = label_lab_after.to(dtype=torch.long)
            input_dict["label_lab"] = label_lab
        if label_xml_after is not None:
            label_xml = label_xml_after.to(dtype=torch.long)
            input_dict["label_xml"] = label_xml
        if ds is not None:
            input_dict.update(ds=ds)
        if tempo_lab_after is not None:
            tempo_lab = tempo_lab_after.to(dtype=torch.long)
            input_dict.update(tempo_lab=tempo_lab)
        if tempo_xml_after is not None:
            tempo_xml = tempo_xml_after.to(dtype=torch.long)
            input_dict.update(tempo_xml=tempo_xml)
        if beat_lab_after is not None:
            beat_lab = beat_lab_after.to(dtype=torch.long)
            input_dict.update(beat_lab=beat_lab)
        if beat_xml_after is not None:
            beat_xml = beat_xml_after.to(dtype=torch.long)
            input_dict.update(beat_xml=beat_xml)
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
