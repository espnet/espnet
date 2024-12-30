# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
# Copyright 2022 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Singing-voice-synthesis ESPnet model."""

import logging

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.espnet_model import ESPnetSVSModel
from espnet2.svs.feats_extract.score_feats_extract import (
    FrameScoreFeats,
    SyllableScoreFeats,
    expand_to_frame,
)
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.svs.net_utils import pad_and_concat
from espnet2.torch_utils.device_funcs import force_gatherable

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetDiscreteRLSVSModel(ESPnetSVSModel):
    """ESPnet model for singing voice synthesis task."""

    def __init__(
        self,
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
        svs: AbsSVS,
        # RL related
        ref_svs: AbsSVS,
        algo: str = "dpo",
        beta: float = 0.1,
        ce_loss_weight: float = 0.0,
        rl_loss_weight: float = 1.0,
        length_norm: bool = False,
        reward_margin: float = 0.0,
        # discrete realted
        discrete_token_layers: int = 1,
    ):
        """Initialize ESPnetSVSModel module."""
        super().__init__(
            text_extract=text_extract,
            feats_extract=feats_extract,
            score_feats_extract=score_feats_extract,
            label_extract=label_extract,
            pitch_extract=pitch_extract,
            ying_extract=ying_extract,
            duration_extract=duration_extract,
            energy_extract=energy_extract,
            normalize=normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            svs=svs,
        )
        self.discrete_token_layers = discrete_token_layers
        self.algo = algo
        self.beta = beta
        self.ce_loss_weight = ce_loss_weight
        self.rl_loss_weight = rl_loss_weight
        self.length_norm = length_norm
        self.reward_margin = reward_margin
        if algo in ["simpo"]:
            del reflm
            self.ref_svs = None
        else:
            # NOTE(Yuxun):no grad in ref_svs
            self.ref_svs = ref_svs 

        if algo in ["simpo"]:
            assert self.length_norm, f"Algo {algo} requires length_normalize"

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
        slur_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        ying: Optional[torch.Tensor] = None,
        ying_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        discrete_token: Optional[torch.Tensor] = None,
        discrete_token_lengths: Optional[torch.Tensor] = None,
        flag_IsValid: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.

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
            slur_lengths (Optional[Tensor]): slur length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor (B, T_frame). - f0 sequence
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            discrete_token (Optional[Tensor]): Discrete token tensor (B, T_frame).
            discrete_token_lengths (Optional[Tensor]): Discrete token length tensor (B,).
            kwargs: "utt_id" is among the input.

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.
        """
        with autocast(False):
            # 1. Extract performacne features (actual features) in frame wise
            #    and normalize
            if self.feats_extract is not None and feats is None:
                # spec feature (frame level) 
                # not used in discrete token
                feats, feats_lengths = self.feats_extract(
                    singing, singing_lengths
                )

            # Extract auxiliary features
            # melody : 128 note pitch
            # duration :
            #   input-> phone-id seqence
            #   output -> frame level(take mode from window) or syllable level

            # algin duration to discrete tokens
            origin_discrete_token_lengths = (
                discrete_token_lengths // self.discrete_token_layers
            )
            for i in range(label.size(0)):
                dur_len = sum(duration_phn[i])
                if origin_discrete_token_lengths[i] > dur_len:
                    delta = origin_discrete_token_lengths[i] - dur_len
                    end = duration_phn_lengths[i] - 1
                    duration_phn[i][end] += delta
                else:  # decrease duration at the end of sequence
                    delta = dur_len - origin_discrete_token_lengths[i]
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
            #            discrete_token = discrete_token[:, : discrete_token_lengths.max()]

            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    input=singing,
                    input_lengths=singing_lengths,
                    feats_lengths=origin_discrete_token_lengths,
                )

            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    singing,
                    singing_lengths,
                    feats_lengths=origin_discrete_token_lengths,
                )

            if self.ying_extract is not None and ying is None:
                ying, ying_lengths = self.ying_extract(
                    singing,
                    singing_lengths,
                    feats_lengths=origin_discrete_token_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

            # 2. Obtain score features in frame/syllabel wise
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
                slur = slur[:, : slur_lengths.max()]
            else:
                raise RuntimeError("Cannot understand score_feats extract type")

        # Make batch for svs inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
            feats=feats,
            feats_lengths=feats_lengths,
            flag_IsValid=flag_IsValid,
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
        melody_lengths = dict()
        if midi_lab is not None:
            midi_lab = midi_lab.to(dtype=torch.long)
            melody.update(lab=midi_lab)
            melody_lengths.update(lab=midi_lab_lengths)
        if midi_score is not None:
            midi_score = midi_score.to(dtype=torch.long)
            melody.update(score=midi_score)
            melody_lengths.update(score=midi_score_lengths)
        batch.update(melody=melody, melody_lengths=melody_lengths)

        # duration
        # NOTE(Yuning): duration = duration_time / time_shift (same as Xiaoice paper)
        duration = dict()
        duration_lengths = dict()
        if duration_lab is not None:
            duration_lab = duration_lab.to(dtype=torch.long)
            duration.update(lab=duration_lab)
            duration_lengths.update(lab=duration_lab_lengths)
        if duration_score is not None:
            duration_phn_score = duration_score.to(dtype=torch.long)
            duration.update(score_phn=duration_phn_score)
            duration_lengths.update(score_phn=duration_score_phn_lengths)
        if duration_score_syb is not None:
            duration_syb_score = duration_score_syb.to(dtype=torch.long)
            duration.update(score_syb=duration_syb_score)
            duration_lengths.update(score_syb=duration_score_syb_lengths)
        batch.update(duration=duration, duration_lengths=duration_lengths)

        if slur is not None:
            batch.update(slur=slur, slur_lengths=slur_lengths)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if self.ying_extract is not None and ying is not None:
            batch.update(ying=ying)
        if self.svs.require_raw_singing:
            batch.update(singing=singing, singing_lengths=singing_lengths)
        if discrete_token is not None:
            batch.update(
                discrete_token=discrete_token,
                discrete_token_lengths=discrete_token_lengths,
                discrete_token_lengths_frame=origin_discrete_token_lengths,
            )
        
        # RL preparation
        if self.ref_svs is not None:
            batch.update(flag_RL=True)

        pos_idx = kwargs["pos_idx"] # [B_pos,T, nq]
        B, T, nq, S = pos_idx.size()
        pos_idx = pos_idx.permute(0, 3, 1, 2).reshape(B * S, T, nq)
        # logging.info(f'pos_idx: {pos_idx.shape}')
        
        neg_idx = kwargs["neg_idx"] # [B_neg, T, nq]
        B, T, nq, S = neg_idx.size()
        neg_idx = neg_idx.permute(0, 3, 1, 2).reshape(B * S, T, nq)
        # logging.info(f'neg_idx: {neg_idx.shape}')

        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        all_idx = pad_and_concat([pos_idx, neg_idx]) # [B_pos + B_neg, T, nq]

        pos_length = kwargs["pos_idx_lengths"] # [B_pos]
        assert len(pos_idx) % len(pos_length) == 0, (len(pos_idx), len(pos_length))
        pos_ratio = int(len(pos_idx) / len(pos_length))
        pos_length = pos_length.repeat_interleave(pos_ratio, dim=0)

        neg_length = kwargs["neg_idx_lengths"] # [B_neg]
        assert len(neg_idx) % len(neg_length) == 0, (len(neg_idx), len(neg_length))
        neg_ratio = int(len(neg_idx) / len(neg_length))
        neg_length = neg_length.repeat_interleave(neg_ratio, dim=0)

        all_length = torch.cat([pos_length, neg_length]) # [B_pos + B_neg]
        assert n_neg % n_pos == 0, (n_neg, n_pos)

        fixed_modules = [
            "phone_encode_layer",
            "midi_encode_layer",
            "duration_encode_layer",
            "encoder",
        ]
        for fix_module_name in fixed_modules:
            for name, module in self.svs.named_modules():
                if name.startswith(fix_module_name + ".") or name == fix_module_name: 
                    module.eval()

        # logging.info(f'batch: {batch}')
        for key, value in batch.items():
            if key == "pitch":
                continue
            if isinstance(value, torch.Tensor):
                logging.info(f'{key}: {value.shape} {value}')
        # for name, module in self.svs.named_modules():
        #     print(f"{name}: training={module.training}")

        _, stats, weight, policy_logits = self.svs(**batch)
        stats["svs_loss"] = stats.pop("loss")

        if self.ref_svs is not None:
            _, _, _, ref_logits = self.ref_svs(**batch)
        else:
            ref_logits = None

        if policy_logits.size() == 3:
            policy_logits = policy_logits.unsequeeze(2) # [B, T, V] -> [B, T, nq, V]
        if ref_logits.size() == 3:
            ref_logits = ref_logits.unsequeeze(2) # [B, T, V] -> [B, T, nq, V]

        loss_rl, stats_rl = self.loss_rl(
            policy_logits,
            ref_logits,
            all_idx,
            all_length,
            n_pos,
        )
        loss = loss_rl
        stats.update(stats_rl)
        stats.update({"loss": loss.detach()})
        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)

        return loss, stats, weight

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
        slur_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        ying: Optional[torch.Tensor] = None,
        ying_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        discrete_token: Optional[torch.Tensor] = None,
        discrete_token_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

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
            ---- duration* is duration in time_shift ----
            duration_phn (Optional[Tensor]): duration tensor (B, T_label).
            duration_phn_lengths (Optional[Tensor]): duration length tensor (B,).
            duration_ruled_phn (Optional[Tensor]): duration tensor (B, T_phone).
            duration_ruled_phn_lengths (Optional[Tensor]): duration length tensor (B,).
            duration_syb (Optional[Tensor]): duration tensor (B, T_syb).
            duration_syb_lengths (Optional[Tensor]): duration length tensor (B,).
            slur (Optional[Tensor]): slur tensor (B, T_slur).
            slur_lengths (Optional[Tensor]): slur length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor (B, T_frame). - f0 sequence
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            discrete_token (Optional[Tensor]): Discrete tokens tensor (B, T_frame)
            discrete_token_lengths (Optional[Tensor]): Discrete tokens lengths tensor (B,)

        Returns:
            Dict[str, Tensor]: Dict of features.
        """
        feats = None

        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(singing, singing_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            feats, feats_lengths = singing, singing_lengths
        # cut length
        """
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
        """
        origin_discrete_token_lengths = (
            discrete_token_lengths // self.discrete_token_layers
        )
        for i in range(label.size(0)):
            dur_len = sum(duration_phn[i])
            if origin_discrete_token_lengths[i] > dur_len:
                delta = origin_discrete_token_lengths[i] - dur_len
                end = duration_phn_lengths[i] - 1
                duration_phn[i][end] += delta
            else:  # decrease duration at the end of sequence
                delta = dur_len - origin_discrete_token_lengths[i]
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
        #        discrete_token = discrete_token[:, : discrete_token_lengths.max()]

        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                input=singing,
                input_lengths=singing_lengths,
                feats_lengths=origin_discrete_token_lengths,
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
        discrete_token: Optional[torch.Tensor] = None,
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
            pitch (Optional[Tensor): Pitch tensor (T_frame).
            energy (Optional[Tensor): Energy tensor.
            discrete_token (Optional[Tensor]): Discrete tokens (T_frame)

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

        origin_discrete_token_lengths = (
            len(discrete_token) // self.discrete_token_layers
        )
        if decode_config["use_teacher_forcing"]:
            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    singing[None],
                    feats_lengths=torch.LongTensor([origin_discrete_token_lengths]),
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
        if discrete_token is not None:
            input_dict.update(discrete_token=discrete_token)

        output_dict = self.svs.inference(**input_dict, **decode_config)
        """
        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)
        """
        return output_dict

    def loss_rl(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        all_idx: torch.Tensor,
        all_length: torch.Tensor,
        n_pos: torch.Tensor,
    ):
        """Compute Reinforcement Learning loss
        
            pos_logits (torch.Tensor): the policy logits of positive
                sequences. (B_pos, T, nq, V)
            ref_logits (torch.Tensor): the policy logits of positive
                sequences. (B_pos, T, nq, V)
            all_idx (torch.Tensor): index of positive + negative. (B_pos + B_neg, T, nq)
            all_length (torch.Tensor): length of logits. (B_pos + B_neg)
            n_pos (torch.Tensor): number of positive. (B_pos,)
        """
        # nq = discrete token layer
        # (1) mask for pad
        mask = make_non_pad_mask(all_length).to(all_length.device)
        pos_ratio = len(all_length) // n_pos - 1

        # (2) logp, summed to utterance level
        # pos policy
        pos_policy_logp = torch.gather(
            policy_logits.log_softmax(-1), dim=3, index=all_idx[:n_pos].unsqueeze(3)
        ).squeeze(3) # [B_pos + B_neg, T, nq]
        pos_policy_logp = (pos_policy_logp * mask[:n_pos]).sum(dim=(1, 2))

        # neg policy
        policy_logits = policy_logits.tile(pos_ratio) # [B_pos -> B_pos + B_neg, T, nq, V]
        neg_policy_logp = torch.gather(
            policy_logits.log_softmax(-1), dim=3, index=all_idx[:n_pos].unsqueeze(3)
        ).squeeze(3) # [B_pos + B_neg, T, nq]
        neg_policy_logp = (neg_policy_logp * mask[:n_pos]).sum(dim=(1, 2))

        if ref_logits is not None:
            # pos ref
            pos_ref_logp = torch.gather(
                ref_logits.log_softmax(-1), dim=3, index=all_idx[n_pos:].unsqueeze(3)
            ).squeeze(3) # [B_pos + B_neg, T, nq]
            pos_ref_logp = (pos_ref_logp * mask[n_pos:]).sum(dim=(1, 2))

            # neg ref
            ref_logits = ref_logits.tile(pos_ratio) # [B_pos -> B_pos + B_neg, T, nq, V]
            neg_ref_logp = torch.gather(
                ref_logits.log_softmax(-1), dim=3, index=all_idx[n_pos:].unsqueeze(3)
            ).squeeze(3)  # [B_pos + B_neg, T, nq]
            neg_ref_logp = (neg_ref_logp * mask[n_pos:]).sum(dim=(1, 2))  # [B_pos + B_neg]
        else:
            pos_ref_logp = torch.zeros_like(pos_policy_logp)
            neg_ref_logp = torch.zeros_like(neg_policy_logp)
        
        # (3) length narmalize
        if self.length_norm:
            nq = all_idx.size(-1)
            pos_policy_logp = pos_policy_logp / all_length[:n_pos] / nq
            neg_policy_logp = neg_policy_logp / all_length[n_pos:] / nq
            pos_ref_logp = pos_ref_logp / all_length[:n_pos] / nq
            neg_ref_logp = neg_ref_logp / all_length[n_pos:] / nq

        # (4) loss computation
        loss_rl, stats = self.compute_loss(
            pos_policy_logp = pos_policy_logp.tile(pos_ratio),
            neg_policy_logp = neg_policy_logp,
            pos_ref_logp = pos_ref_logp.tile(pos_ratio),
            neg_ref_logp = neg_ref_logp,
        )
        return loss_rl.mean(), stats

    def compute_loss(
        self,
        pos_policy_logp: torch.Tensor,
        neg_policy_logp: torch.Tensor,
        pos_ref_logp: torch.Tensor,
        neg_ref_logp: torch.Tensor,
    ):
        """Compute exactly the DPO-series loss"""
        logits = (pos_policy_logp - neg_policy_logp) - (pos_ref_logp - neg_ref_logp)

        if self.algo == "dpo":
            loss = -F.logsigmoid(logits * self.beta)

        elif self.algo == "simpo":
            loss = -F.logsigmoid(logits * self.beta - self.reward_margin)

        else:
            raise NotImplementedError(f"{self.algo} is not supported yet")

        pos_reward = pos_policy_logp - pos_ref_logp
        neg_reward = neg_policy_logp - neg_ref_logp
        acc = (pos_reward > neg_reward).float()

        stats = {
            "loss_rl": loss,
            "pos_reward": pos_reward,
            "neg_reward": neg_reward,
            "reward_gap": pos_reward - neg_reward,
            "reward_acc": acc,
            "pos_policy_logp": pos_policy_logp,
            "neg_policy_logp": neg_policy_logp,
            "pos_ref_logp": pos_ref_logp,
            "neg_ref_logp": neg_ref_logp,
        }
        stats = {k: v.detach().mean() for k, v in stats.items()}

        return loss, stats
