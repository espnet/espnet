from itertools import chain
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


SINGLE_SPK=0
MULTI_SPK=1


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


        # self.end2end_train = False
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
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)]
            if f"speech_ref{spk + 1}" in kwargs
            else None
            for spk in range(self.num_spk)
        ]
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
        text_ref = [
            torch.cat(
                [
                    ref,
                    torch.ones(batch_size, text_length_max, dtype=ref.dtype).to(
                        ref.device
                    )
                    * self.idx_blank,
                ],
                dim=1,
            )[:, :text_length_max]
            for ref in text_ref
        ]

        # 0. Enhancement
        if "utt2category" in kwargs:
            utt2category = kwargs["utt2category"][0]
        else:
            utt2category = 0
        # print(utt2category)

        # make sure the speech_pre is the raw waveform with same size.
        # if len(text_ref) == 1 or all(
        #     [tr.equal(text_ref[0]) for tr in text_ref[1:]]
        # ):
        # if False:
        if utt2category.int() == SINGLE_SPK:
            # TODO(Jing): find a better way to locate single-spk set
            # single-speaker case
            speech_pre_all = (
                speech_mix
                if speech_mix.dim() == 2
                else speech_mix[..., self.ref_channel]
            )
            speech_pre_lengths = speech_mix_lengths
            text_ref_all, text_ref_lengths = text_ref[0], text_ref_lengths[0]
            perm = True
            loss_enh = None
            n_speaker_asr = 1
        elif utt2category.int() == MULTI_SPK:
            loss_enh, perm, speech_pre, speech_pre_lengths = self.enh_subclass.forward(
                speech_mix,
                speech_mix_lengths,
                asr_integration=True,
                speech_ref=speech_ref,
            )
            # speech_pre: List[bs,T] --> (bs,num_spk,T)
            speech_pre = torch.stack(speech_pre, dim=1)
            if speech_pre[:, 0].dim() == speech_mix.dim():
                # single-channel input
                assert speech_pre[:, 0].shape == speech_mix.shape, (
                    speech_pre[:, 0].shape,
                    speech_mix.shape,
                )
                speech_frame_length = speech_mix.size(-1)
            else:
                # multi-channel input
                assert speech_pre[:, 0].shape == speech_mix[..., 0].shape, (
                    speech_pre[:, 0].shape,
                    speech_mix.shape,
                )
                speech_frame_length = speech_mix.size(-2)

            if not self.end2end_train:
                # if the FrontEnd and ASR are trained independetly
                # use the speech_ref to train asr
                speech_pre = torch.stack(speech_ref, dim=1)

            # Pack the separated speakers into the ASR part.
            speech_pre_all = (
                speech_pre.transpose(0, 1)
                .contiguous()
                .view(-1, speech_frame_length)
            )  # (N_spk*B, T)
            speech_pre_lengths = torch.stack(
                [speech_mix_lengths, speech_mix_lengths], dim=1
            ).view(-1)
            text_ref_all = torch.stack(text_ref, dim=1).view(
                batch_size * len(text_ref), -1
            )
            text_ref_lengths = torch.stack(text_ref_lengths, dim=1).view(-1)
            n_speaker_asr = 1 if self.cal_enh_loss else self.num_spk


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

    def _permutation_loss(self, ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            loss: torch.Tensor: (batch)
            perm: list[(num_spk)]
        """
        num_spk = len(ref)

        losses = torch.stack(
            [
                torch.stack([criterion(ref[r], inf[h]) for r in range(num_spk)], dim=1)
                for h in range(num_spk)
            ],
            dim=2,
        )  # (B, n_ref, n_hyp)
        perm_detail, min_loss = self.permutation_invariant_training(losses)

        return min_loss.mean(), perm_detail

    def permutation_invariant_training(self, losses: torch.Tensor):
        """Compute  PIT loss.

        Args:
            losses (torch.Tensor): (batch, nref, nhyp)
        Returns:
            perm: list: (batch, n_spk)
            loss: torch.Tensor: (batch)
        """
        hyp_perm, min_perm_loss = [], []
        losses_cpu = losses.data.cpu()
        for b, b_loss in enumerate(losses_cpu):
            # hungarian algorithm
            try:
                row_ind, col_ind = linear_sum_assignment(b_loss)
            except ValueError as err:
                if str(err) == "cost matrix is infeasible":
                    # random assignment since the cost is always inf
                    col_ind = np.array([0, 1])
                    min_perm_loss.append(torch.mean(losses[b, col_ind, col_ind]))
                    hyp_perm.append(col_ind)
                    continue
                else:
                    raise

            min_perm_loss.append(torch.mean(losses[b, row_ind, col_ind]))
            hyp_perm.append(col_ind)

        return hyp_perm, torch.stack(min_perm_loss)
