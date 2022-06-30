from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from typeguard import check_argument_types
"""
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_muskit_model import AbsMuskitModel
from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.feats_extract.abs_feats_extract import AbsFeatsExtract

from espnet2.svs.feats_extract.score_feats_extract import FrameScoreFeats
from espnet2.svs.feats_extract.score_feats_extract import SyllableScoreFeats
"""
from espnet2.torch_utils.nets_utils import pad_list

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class MuskitSVSModel(AbsMuskitModel):
    """Muskit model for singing voice synthesis task."""

    def __init__(
        self,
        text_extract: Optional[AbsFeatsExtract],
        feats_extract: Optional[AbsFeatsExtract],
        score_feats_extract: Optional[AbsFeatsExtract],
        durations_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        tempo_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        # text_normalize: Optional[AbsNormalize and InversibleInterface],
        # durations_normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        # tempo_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        svs: AbsSVS,
    ):
        """Initialize MuskitSVSModel module."""
        assert check_argument_types()
        super().__init__()
        self.text_extract = text_extract
        self.feats_extract = feats_extract
        self.score_feats_extract = score_feats_extract
        self.durations_extract = durations_extract
        self.pitch_extract = pitch_extract
        self.tempo_extract = tempo_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        # self.text_normalize = text_normalize
        # self.durations_normalize = durations_normalize
        self.pitch_normalize = pitch_normalize
        # self.tempo_normalize = tempo_normalize
        self.energy_normalize = energy_normalize
        self.svs = svs

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.
        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            duration (Optional[Tensor]): Duration tensor. - phone id sequence
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            score (Optional[Tensor]): Duration tensor.
            score_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.
        """
        with autocast(False):
            # if self.text_extract is not None and text is None:
            #     text, text_lengths = self.text_extract(
            #         input=text,
            #         input_lengths=text_lengths,
            #     )
            # Extract features
            # logging.info(f'singing.shape={singing.shape}, singing_lengths.shape={singing_lengths.shape}')
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
            #   input-> phone-id seqence | output -> frame level(取众数 from window) or syllable level
            ds = None
            if isinstance(self.score_feats_extract, FrameScoreFeats):
                (
                    label,
                    label_lengths,
                    score,
                    score_lengths,
                    tempo,
                    tempo_lengths,
                ) = self.score_feats_extract(
                    durations=durations.unsqueeze(-1),
                    durations_lengths=durations_lengths,
                    score=score.unsqueeze(-1),
                    score_lengths=score_lengths,
                    tempo=tempo.unsqueeze(-1),
                    tempo_lengths=tempo_lengths,
                )

                label = label[:, : label_lengths.max()]  # for data-parallel

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
                for i, _ in enumerate(label_lengths):
                    _phone = label[i, : label_lengths[i]]

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

            elif isinstance(self.score_feats_extract, SyllableScoreFeats):
                extractMethod_frame = FrameScoreFeats(
                    fs=self.score_feats_extract.fs,
                    n_fft=self.score_feats_extract.n_fft,
                    win_length=self.score_feats_extract.win_length,
                    hop_length=self.score_feats_extract.hop_length,
                    window=self.score_feats_extract.window,
                    center=self.score_feats_extract.center,
                )

                # logging.info(f"extractMethod_frame: {extractMethod_frame}")

                (
                    labelFrame,
                    labelFrame_lengths,
                    scoreFrame,
                    scoreFrame_lengths,
                    tempoFrame,
                    tempoFrame_lengths,
                ) = extractMethod_frame(
                    durations=durations.unsqueeze(-1),
                    durations_lengths=durations_lengths,
                    score=score.unsqueeze(-1),
                    score_lengths=score_lengths,
                    tempo=tempo.unsqueeze(-1),
                    tempo_lengths=tempo_lengths,
                )

                labelFrame = labelFrame[
                    :, : labelFrame_lengths.max()
                ]  # for data-parallel
                scoreFrame = scoreFrame[
                    :, : scoreFrame_lengths.max()
                ]  # for data-parallel

                # Extract Syllable Level label, score, tempo information from Frame Level
                (
                    label,
                    label_lengths,
                    score,
                    score_lengths,
                    tempo,
                    tempo_lengths,
                ) = self.score_feats_extract(
                    durations=labelFrame,
                    durations_lengths=labelFrame_lengths,
                    score=scoreFrame,
                    score_lengths=scoreFrame_lengths,
                    tempo=tempoFrame,
                    tempo_lengths=tempoFrame_lengths,
                )

                # calculate durations, represent syllable encoder outputs to feats mapping
                # Syllable Level duration info needs phone & midi
                ds = []
                for i, _ in enumerate(labelFrame_lengths):
                    assert labelFrame_lengths[i] == scoreFrame_lengths[i]
                    assert label_lengths[i] == score_lengths[i]

                    frame_length = labelFrame_lengths[i]
                    _phoneFrame = labelFrame[i, :frame_length]
                    _midiFrame = scoreFrame[i, :frame_length]

                    # Clean _phoneFrame & _midiFrame
                    for index in range(frame_length):
                        if _phoneFrame[index] == 0 and _midiFrame[index] == 0:
                            frame_length -= 1
                            feats_lengths[i] -= 1

                    syllable_length = label_lengths[i]
                    _phoneSyllable = label[i, :syllable_length]
                    _midiSyllable = score[i, :syllable_length]

                    # logging.info(f"_phoneFrame: {_phoneFrame}, _midiFrame: {_midiFrame}")
                    # logging.info(f"_phoneSyllable: {_phoneSyllable}, _midiSyllable: {_midiSyllable}, _tempoSyllable: {tempo[i]}")

                    start_index = 0
                    ds_tmp = []
                    flag_finish = 0
                    for index in range(syllable_length):
                        _findPhone = _phoneSyllable[index]
                        _findMidi = _midiSyllable[index]
                        _length = 0
                        if flag_finish == 1:
                            # Fix error in _phoneSyllable & _midiSyllable
                            label[i, index] = 0
                            score[i, index] = 0
                            tempo[i, index] = 0
                            label_lengths[i] -= 1
                            score_lengths[i] -= 1
                            tempo_lengths[i] -= 1
                        else:
                            for indexFrame in range(start_index, frame_length):
                                if (
                                    _phoneFrame[indexFrame] == _findPhone
                                    and _midiFrame[indexFrame] == _findMidi
                                ):
                                    _length += 1
                                else:
                                    # logging.info(f"_findPhone: {_findPhone}, _findMidi: {_findMidi}, _length: {_length}")
                                    ds_tmp.append(_length)
                                    start_index = indexFrame
                                    break
                                if indexFrame == frame_length - 1:
                                    # logging.info(f"_findPhone: {_findPhone}, _findMidi: {_findMidi}, _length: {_length}")
                                    flag_finish = 1
                                    ds_tmp.append(_length)
                                    start_index = indexFrame
                                    # logging.info("Finish")
                                    break

                    # logging.info(f"ds_tmp: {ds_tmp}, sum(ds_tmp): {sum(ds_tmp)}, frame_length: {frame_length}, feats_lengths[i]: {feats_lengths[i]}")
                    assert (
                        sum(ds_tmp) == frame_length and sum(ds_tmp) == feats_lengths[i]
                    )

                    ds.append(torch.tensor(ds_tmp))
                ds = pad_list(ds, pad_value=0).to(label.device)

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
                    durations=durations,
                    durations_lengths=durations_lengths,
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
        if label is not None:
            label = label.to(dtype=torch.long)
            batch.update(label=label, label_lengths=label_lengths)
        if score is not None and pitch is None:
            score = score.to(dtype=torch.long)
            batch.update(midi=score, midi_lengths=score_lengths)
        if tempo is not None:
            tempo = tempo.to(dtype=torch.long)
            batch.update(tempo=tempo, tempo_lengths=tempo_lengths)
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
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.
        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            durations (Optional[Tensor): Duration tensor.
            durations_lengths (Optional[Tensor): Duration length tensor (B,).
            score (Optional[Tensor): Duration tensor.
            score_lengths (Optional[Tensor): Duration length tensor (B,).
            pitch (Optional[Tensor): Pitch tensor.
            pitch_lengths (Optional[Tensor): Pitch length tensor (B,).
            tempo (Optional[Tensor): Tempo tensor.
            tempo_lengths (Optional[Tensor): Tempo length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
        Returns:
            Dict[str, Tensor]: Dict of features.
        """
        # feature extraction
        # if self.text_extract is not None:
        #     text, text_lengths = self.text_extract(
        #         input=text,
        #         input_lengths=text_lengths,
        #     )
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(singing, singing_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            feats, feats_lengths = singing, singing_lengths
        if self.score_feats_extract is not None:
            (
                durations,
                durations_lengths,
                score,
                score_lengths,
                tempo,
                tempo_lengths,
            ) = self.score_feats_extract(
                durations=durations.unsqueeze(-1),
                durations_lengths=durations_lengths,
                score=score.unsqueeze(-1),
                score_lengths=score_lengths,
                tempo=tempo.unsqueeze(-1),
                tempo_lengths=tempo_lengths,
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
                durations=durations,
                durations_lengths=durations_lengths,
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
        durations: torch.Tensor,
        score: torch.Tensor,
        singing: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
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
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.
        Returns:
            Dict[str, Tensor]: Dict of outputs.
        """
        durations_lengths = torch.tensor([len(durations)])
        score_lengths = torch.tensor([len(score)])
        tempo_lengths = torch.tensor([len(tempo)])

        assert durations_lengths == score_lengths and durations_lengths == tempo_lengths

        # unsqueeze of singing must be here, or it'll cause error in the return dim of STFT
        text = text.unsqueeze(0)  # for data-parallel
        durations = durations.unsqueeze(0)  # for data-parallel
        score = score.unsqueeze(0)  # for data-parallel
        tempo = tempo.unsqueeze(0)  # for data-parallel

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
                label,
                label_lengths,
                score,
                score_lengths,
                tempo,
                tempo_lengths,
            ) = self.score_feats_extract(
                durations=durations.unsqueeze(-1),
                durations_lengths=durations_lengths,
                score=score.unsqueeze(-1),
                score_lengths=score_lengths,
                tempo=tempo.unsqueeze(-1),
                tempo_lengths=tempo_lengths,
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
                _phone = label[i]

                _output, counts = torch.unique_consecutive(_phone, return_counts=True)

                _text_cal.append(_output)
                _text_length_cal.append(len(_output))
                ds.append(counts)
            ds = pad_list(ds, pad_value=0).to(text.device)
            text = pad_list(_text_cal, pad_value=0).to(text.device, dtype=torch.long)
            text_lengths = torch.tensor(_text_length_cal).to(text.device)

        elif isinstance(self.score_feats_extract, SyllableScoreFeats):
            extractMethod_frame = FrameScoreFeats(
                fs=self.score_feats_extract.fs,
                n_fft=self.score_feats_extract.n_fft,
                win_length=self.score_feats_extract.win_length,
                hop_length=self.score_feats_extract.hop_length,
                window=self.score_feats_extract.window,
                center=self.score_feats_extract.center,
            )

            # logging.info(f"extractMethod_frame: {extractMethod_frame}")

            (
                labelFrame,
                labelFrame_lengths,
                scoreFrame,
                scoreFrame_lengths,
                tempoFrame,
                tempoFrame_lengths,
            ) = extractMethod_frame(
                durations=durations.unsqueeze(-1),
                durations_lengths=durations_lengths,
                score=score.unsqueeze(-1),
                score_lengths=score_lengths,
                tempo=tempo.unsqueeze(-1),
                tempo_lengths=tempo_lengths,
            )

            labelFrame = labelFrame[:, : labelFrame_lengths.max()]  # for data-parallel
            scoreFrame = scoreFrame[:, : scoreFrame_lengths.max()]  # for data-parallel

            # Extract Syllable Level label, score, tempo information from Frame Level
            (
                label,
                label_lengths,
                score,
                score_lengths,
                tempo,
                tempo_lengths,
            ) = self.score_feats_extract(
                durations=labelFrame,
                durations_lengths=labelFrame_lengths,
                score=scoreFrame,
                score_lengths=scoreFrame_lengths,
                tempo=tempoFrame,
                tempo_lengths=tempoFrame_lengths,
            )

            # calculate durations, represent syllable encoder outputs to feats mapping
            # Syllable Level duration info needs phone & midi
            ds = []
            for i, _ in enumerate(labelFrame_lengths):
                assert labelFrame_lengths[i] == scoreFrame_lengths[i]
                assert label_lengths[i] == score_lengths[i]

                frame_length = labelFrame_lengths[i]
                _phoneFrame = labelFrame[i, :frame_length]
                _midiFrame = scoreFrame[i, :frame_length]

                # Clean _phoneFrame & _midiFrame
                for index in range(frame_length):
                    if _phoneFrame[index] == 0 and _midiFrame[index] == 0:
                        frame_length -= 1
                        # feats_lengths[i] -= 1

                syllable_length = label_lengths[i]
                _phoneSyllable = label[i, :syllable_length]
                _midiSyllable = score[i, :syllable_length]

                # logging.info(f"_phoneFrame: {_phoneFrame}, _midiFrame: {_midiFrame}")
                # logging.info(f"_phoneSyllable: {_phoneSyllable}, _midiSyllable: {_midiSyllable}, _tempoSyllable: {tempo[i]}")

                start_index = 0
                ds_tmp = []
                flag_finish = 0
                for index in range(syllable_length):
                    _findPhone = _phoneSyllable[index]
                    _findMidi = _midiSyllable[index]
                    _length = 0
                    if flag_finish == 1:
                        # Fix error in _phoneSyllable & _midiSyllable
                        label[i, index] = 0
                        score[i, index] = 0
                        tempo[i, index] = 0
                        label_lengths[i] -= 1
                        score_lengths[i] -= 1
                        tempo_lengths[i] -= 1
                    else:
                        for indexFrame in range(start_index, frame_length):
                            if (
                                _phoneFrame[indexFrame] == _findPhone
                                and _midiFrame[indexFrame] == _findMidi
                            ):
                                _length += 1
                            else:
                                # logging.info(f"_findPhone: {_findPhone}, _findMidi: {_findMidi}, _length: {_length}")
                                ds_tmp.append(_length)
                                start_index = indexFrame
                                break
                            if indexFrame == frame_length - 1:
                                # logging.info(f"_findPhone: {_findPhone}, _findMidi: {_findMidi}, _length: {_length}")
                                flag_finish = 1
                                ds_tmp.append(_length)
                                start_index = indexFrame
                                # logging.info("Finish")
                                break

                logging.info(
                    f"ds_tmp: {ds_tmp}, sum(ds_tmp): {sum(ds_tmp)}, frame_length: {frame_length}"
                )
                assert sum(ds_tmp) == frame_length

                ds.append(torch.tensor(ds_tmp))
            ds = pad_list(ds, pad_value=0).to(label.device)

        input_dict = dict(text=text)

        if score is not None and pitch is None:
            score = score.to(dtype=torch.long)
            input_dict["midi"] = score
        if durations is not None:
            label = label.to(dtype=torch.long)
            input_dict["label"] = label
        if ds is not None:
            input_dict.update(ds=ds)
        if tempo is not None:
            tempo = tempo.to(dtype=torch.long)
            input_dict.update(tempo=tempo)
        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        # output_dict = self.svs.inference(**input_dict, **decode_config)
        outs, probs, att_ws = self.svs.inference(**input_dict)

        if self.normalize is not None:
            # NOTE: normalize.inverse is in-place operation
            outs_denorm = self.normalize.inverse(outs.clone()[None])[0][0]
        else:
            outs_denorm = outs

        return outs, outs_denorm, probs, 