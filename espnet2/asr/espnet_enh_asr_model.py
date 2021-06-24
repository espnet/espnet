from distutils.version import LooseVersion
from itertools import chain
import random
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.category import UttCategory


is_torch_1_8_plus = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")


def fliter_attrs(a, b):
    a_attr = [attr for attr in a if attr[:2] != "__" and attr[-2:] != "__"]
    b_attr = [attr for attr in b if attr[:2] != "__" and attr[-2:] != "__"]
    a_attr_ = [i for i in a_attr if i not in b_attr]
    b_attr_ = [i for i in b_attr if i not in a_attr]
    return a_attr_, b_attr_


class ESPnetEnhASRModel(AbsESPnetModel):
    """Enhancement frontend with CTC-attention hybrid Encoder-Decoder model."""

    def __init__(
        self,
        enh_model: Optional[ESPnetEnhancementModel],
        asr_model: Optional[ESPnetASRModel],
        enh_weight: float = 0.5,
        cal_enh_loss: bool = True,
        end2end_train: bool = True,
        total_loss_scale: float = 1,
        # for multi-condition training with real data when num_spk == 1
        enh_real_prob: float = 1.0,
    ):
        assert check_argument_types()
        assert 0.0 <= asr_model.ctc_weight <= 1.0, asr_model.ctc_weight
        assert 0.0 <= enh_weight <= 1.0, asr_model.ctc_weight
        assert asr_model.rnnt_decoder is None, "Not implemented"
        super().__init__()

        self.enh_subclass = enh_model
        self.asr_subclass = asr_model
        self.enh_weight = enh_weight
        self.cal_enh_loss = cal_enh_loss
        self.total_loss_scale = total_loss_scale

        # TODO(Jing): find out the -1 or 0 here
        # self.idx_blank = token_list.index(sym_blank) # 0
        self.idx_blank = -1
        self.num_spk = enh_model.num_spk
        if self.num_spk > 1:
            assert (
                asr_model.ctc_weight != 0.0 or cal_enh_loss
            )  # need at least one to cal PIT permutation

        self.end2end_train = end2end_train
        self.enh_attr = dir(enh_model)
        self.asr_attr = dir(asr_model)

        # Note(Jing): self delegation from the enh and asr sub-modules
        # fliter the specific attr for each subclass
        self.enh_attr, self.asr_attr = fliter_attrs(self.enh_attr, self.asr_attr)
        for arr in self.enh_attr:
            setattr(self, arr, getattr(self.enh_subclass, arr))
        for arr in self.asr_attr:
            setattr(self, arr, getattr(self.asr_subclass, arr))

        # for multi-condition training with real data when self.num_spk == 1
        # if < 1.0, feed the real data only to ASR with probability `enh_real_prob`
        self.enh_real_prob = enh_real_prob

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Enhancement + Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        text_ref = [kwargs["text_ref{}".format(spk + 1)] for spk in range(self.num_spk)]
        text_ref_lengths = [
            kwargs["text_ref{}_lengths".format(spk + 1)] for spk in range(self.num_spk)
        ]

        assert all(ref_lengths.dim() == 1 for ref_lengths in text_ref_lengths), (
            ref_lengths.shape for ref_lengths in text_ref_lengths
        )
        # Check that batch_size is unified
        batch_size = speech_mix.shape[0]
        assert batch_size == speech_mix_lengths.shape[0], (
            speech_mix.shape,
            speech_mix_lengths.shape,
        )
        assert all(
            it.shape[0] == batch_size for it in chain(text_ref, text_ref_lengths)
        ), (
            speech_mix.shape,
            (ref.shape for ref in text_ref),
            (ref_lengths.shape for ref_lengths in text_ref_lengths),
        )

        # for data-parallel
        text_length_max = max(ref_lengths.max() for ref_lengths in text_ref_lengths)
        # pad text sequences of different speakers to the same length
        text_ref = [
            torch.cat(
                [ref, ref.new_full((batch_size, text_length_max), self.idx_blank)],
                dim=1,
            )[:, :text_length_max]
            for ref in text_ref
        ]

        # 0. Enhancement
        if "utt2category" in kwargs:
            utt2category = kwargs["utt2category"][0].int()
        else:
            utt2category = UttCategory.SIMU_DATA

        if utt2category == UttCategory.SIMU_DATA or (
            utt2category == UttCategory.REAL_1SPEAKER and self.num_spk == 1
        ):
            is_simu_data = utt2category == UttCategory.SIMU_DATA
            if is_simu_data or random.random() <= self.enh_real_prob:
                # 1. For simulated single-/multi-speaker data (SIMU_DATA),
                #    feed it to Enh FrontEnd and calculate loss_enh
                # 2. For single-speaker real data (REAL_1SPEAKER),
                #    feed it to Enh FrontEnd but without calculating loss_enh
                #    with some probability for end-to-end SE and ASR
                loss_enh, perm, speech_pre, speech_pre_lengths = self.forward_enh(
                    speech_mix,
                    speech_mix_lengths,
                    cal_loss=is_simu_data,
                    **kwargs,
                )

                # speech_pre: List[(Batch, T)] --> (Batch, num_spk, T)
                speech_pre = torch.stack(speech_pre, dim=1)
                speech_frame_length = speech_mix.size(1)
                if speech_pre[:, 0].dim() == speech_mix.dim():  # single-channel input
                    shape_tmp = speech_mix.shape
                else:  # multi-channel input
                    shape_tmp = speech_mix[..., 0].shape
                assert speech_pre[:, 0].shape == shape_tmp, (
                    speech_pre[:, 0].shape,
                    speech_mix.shape,
                )

                if is_simu_data and not self.end2end_train:
                    # if the FrontEnd and ASR are trained independetly
                    # use the speech_ref to train ASR (only for SIMU_DATA)
                    speech_pre = torch.stack(
                        [
                            kwargs["speech_ref{}".format(spk + 1)]
                            if f"speech_ref{spk + 1}" in kwargs
                            else None
                            for spk in range(self.num_spk)
                        ],
                        dim=1,
                    )

                # Pack the separated speakers into the ASR part.
                # (Batch, num_spk, T) -> (num_spk * Batch, T)
                speech_pre_all = (
                    speech_pre.transpose(0, 1)
                    .contiguous()
                    .view(-1, speech_frame_length)
                )
                # (num_spk * Batch,)
                speech_pre_lengths = torch.stack(
                    [speech_mix_lengths for _ in range(self.num_spk)], dim=1
                ).view(-1)
                text_ref_all = torch.stack(text_ref, dim=1).view(
                    batch_size * len(text_ref), -1
                )
                text_ref_lengths = torch.stack(text_ref_lengths, dim=1).view(-1)
                n_speaker_asr = 1 if self.cal_enh_loss else self.num_spk
            else:
                # 2. with some probability, bypass the Enh Frontend, and feed the
                # real 1-spk data directly to the ASR backend
                speech_pre_all = (
                    speech_mix
                    if speech_mix.dim() == 2
                    else speech_mix[..., self.ref_channel]
                )
                speech_pre_lengths = speech_mix_lengths
                text_ref_all, text_ref_lengths = text_ref[0], text_ref_lengths[0]
                loss_enh, perm = None, None
                n_speaker_asr = 1

        elif utt2category in (UttCategory.CLEAN_1SPEAKER, UttCategory.REAL_1SPEAKER):
            # single-speaker clean/real data, only for ASR
            speech_pre_all = (
                speech_mix
                if speech_mix.dim() == 2
                else speech_mix[..., self.ref_channel]
            )
            speech_pre_lengths = speech_mix_lengths
            text_ref_all, text_ref_lengths = text_ref[0], text_ref_lengths[0]
            loss_enh, perm = None, None
            n_speaker_asr = 1
        else:
            raise ValueError("Unsupported category: %s" % utt2category)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech_pre_all, speech_pre_lengths)

        # 2a. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            if n_speaker_asr == 1 or (
                self.cal_enh_loss and perm is not None
            ):  # No permutation is required
                assert n_speaker_asr == 1
                loss_ctc, cer_ctc, _, _, = self._calc_ctc_loss_with_spk(
                    encoder_out,
                    encoder_out_lens,
                    text_ref_all,
                    text_ref_lengths,
                    n_speakers=n_speaker_asr,
                )
            else:  # Permutation is determined by CTC
                assert n_speaker_asr > 1
                (
                    loss_ctc,
                    cer_ctc,
                    encoder_out,
                    encoder_out_lens,
                ) = self._calc_ctc_loss_with_spk(
                    encoder_out,
                    encoder_out_lens,
                    text_ref_all,
                    text_ref_lengths,
                    n_speakers=self.num_spk,
                )

        # 2b. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        if self.ctc_weight == 0.0:
            loss_asr = loss_att
        elif self.ctc_weight == 1.0:
            loss_asr = loss_ctc
        else:
            loss_asr = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.enh_weight == 0.0 or not self.cal_enh_loss or loss_enh is None:
            loss_enh = None
            loss = loss_asr
        else:
            loss = self.total_loss_scale * (
                (1 - self.enh_weight) * loss_asr + self.enh_weight * loss_enh
            )

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_enh=loss_enh.detach() if loss_enh is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def forward_enh(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        cal_loss: bool,
        **kwargs,
    ):
        """enh forward with or without loss calculation

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            cal_loss: Can set to False only if used in joint SE and ASR (enh_asr)
        [Optional Args]:
            speech_ref: speech references for all speakers List[torch.Tensor]
            noise_ref1: denoising reference for noise 1 (Batchm, samples [, channels])
            ...
            dereverb_ref1: dereverberation reference for speaker 1
                (Batchm, samples [, channels])
            dereverb_ref2: dereverberation reference for speaker 2
                (Batchm, samples [, channels])
            ...
        Returns:
            loss: scalar
            perm: (Batch,)
            speech_pre: List[torch.Tensor(Batch, samples)]
            out_lengths: (Batch,)
        """
        (
            speech_mix,
            speech_lengths,
            speech_ref,
            noise_ref,
            dereverb_speech_ref,
        ) = self.enh_subclass._prepare_data(speech_mix, speech_mix_lengths, kwargs)

        if cal_loss:
            loss, speech_pre, _, out_lengths, perm, _ = self.enh_subclass._compute_loss(
                speech_mix,
                speech_lengths,
                speech_ref,
                dereverb_speech_ref=dereverb_speech_ref,
                noise_ref=noise_ref,
            )

            # make sure speech_pre is waveform
            if isinstance(speech_pre[0], ComplexTensor) or (
                is_torch_1_8_plus and torch.is_complex(speech_pre[0])
            ):
                # speech_pre: list[(batch, sample)]
                speech_pre = [
                    self.enh_subclass.decoder(ps, speech_lengths)[0]
                    for ps in speech_pre
                ]
            assert speech_pre[0].dim() == 2, speech_pre[0].dim()

            # resort the prediction wav with the perm from enh_loss
            speech_pre = ESPnetEnhancementModel.sort_by_perm(speech_pre, perm)

        else:
            loss, perm = None, None
            feature_mix, flens = self.enh_subclass.encoder(speech_mix, speech_lengths)
            feature_pre, flens, others = self.enh_subclass.separator(feature_mix, flens)
            speech_pre = [
                self.enh_subclass.decoder(ps, speech_lengths)[0] for ps in feature_pre
            ]

        return loss, perm, speech_pre, speech_lengths

    def _calc_ctc_loss_with_spk(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        n_speakers: int = 1,
    ):
        # Calc CTC loss
        if n_speakers == 1:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
            # loss_ctc = loss_ctc.masked_fill(torch.isinf(loss_ctc), 0)
            loss_ctc = loss_ctc.mean()
        else:
            encoder_out, encoder_out_lens, ys_pad, ys_pad_lens = (
                torch.chunk(encoder_out, n_speakers, dim=0),
                torch.chunk(encoder_out_lens, n_speakers, dim=0),
                torch.chunk(ys_pad, n_speakers, dim=0),
                torch.chunk(ys_pad_lens, n_speakers, dim=0),
            )
            batch_size = encoder_out[0].size(0)
            loss_ctc = torch.stack(
                [
                    torch.stack(
                        [
                            self.ctc(
                                encoder_out[h],
                                encoder_out_lens[h],
                                ys_pad[r],
                                ys_pad_lens[r],
                            )
                            for r in range(n_speakers)
                        ],
                        dim=1,
                    )
                    for h in range(n_speakers)
                ],
                dim=2,
            )  # (B, n_ref, n_hyp)
            # loss_ctc = loss_ctc.masked_fill(torch.isinf(loss_ctc), 0)
            perm_detail, min_loss_ctc = self.permutation_invariant_training(loss_ctc)
            loss_ctc = min_loss_ctc.mean()
            # permutate the encoder_out
            encoder_out, encoder_out_lens = (
                torch.stack(encoder_out, dim=1),
                torch.stack(encoder_out_lens, dim=1),
            )  # (B, n_spk, T, D)
            for b in range(batch_size):
                encoder_out[b] = encoder_out[b, perm_detail[b]]
                encoder_out_lens[b] = encoder_out_lens[b, perm_detail[b]]
            encoder_out = torch.cat(
                [encoder_out[:, i] for i in range(n_speakers)], dim=0
            )
            encoder_out_lens = torch.cat(
                [encoder_out_lens[:, i] for i in range(n_speakers)], dim=0
            )
            ys_pad = torch.cat(ys_pad, dim=0)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc, encoder_out, encoder_out_lens
