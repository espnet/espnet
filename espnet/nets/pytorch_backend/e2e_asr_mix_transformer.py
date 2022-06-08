#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Transformer speech recognition model for single-channel multi-speaker mixture speech.

It is a fusion of `e2e_asr_mix.py` and `e2e_asr_transformer.py`. Refer to:
    https://arxiv.org/pdf/2002.03921.pdf
1. The Transformer-based Encoder now consists of three stages:
     (a): Enc_mix: encoding input mixture speech;
     (b): Enc_SD: separating mixed speech representations;
     (c): Enc_rec: transforming each separated speech representation.
2. PIT is used in CTC to determine the permutation with minimum loss.
"""

import logging
import math
from argparse import Namespace

import numpy
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_mix import E2E as E2EASRMIX
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2EASR
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.encoder_mix import EncoderMix
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask, target_mask


class E2E(E2EASR, ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2EASR.add_arguments(parser)
        E2EASRMIX.encoder_mix_add_arguments(parser)
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__(idim, odim, args, ignore_id=-1)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = EncoderMix(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks_sd=args.elayers_sd,
            num_blocks_rec=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            num_spkrs=args.num_spkrs,
        )

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False
            )
        else:
            self.ctc = None

        self.num_spkrs = args.num_spkrs
        self.pit = PIT(self.num_spkrs)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences
                                    (B, num_spkrs, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)  # list: speaker differentiate
        self.hs_pad = hs_pad

        # 2. ctc
        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        assert self.mtlalpha > 0.0
        batch_size = xs_pad.size(0)
        ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
        hs_len = [hs_mask[i].view(batch_size, -1).sum(1) for i in range(self.num_spkrs)]
        loss_ctc_perm = torch.stack(
            [
                self.ctc(
                    hs_pad[i // self.num_spkrs].view(batch_size, -1, self.adim),
                    hs_len[i // self.num_spkrs],
                    ys_pad[i % self.num_spkrs],
                )
                for i in range(self.num_spkrs**2)
            ],
            dim=1,
        )  # (B, num_spkrs^2)
        loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)
        logging.info("ctc loss:" + str(float(loss_ctc)))

        # Permute the labels according to loss
        for b in range(batch_size):  # B
            ys_pad[:, b] = ys_pad[min_perm[b], b]  # (num_spkrs, B, Lmax)
        ys_out_len = [
            float(torch.sum(ys_pad[i] != self.ignore_id)) for i in range(self.num_spkrs)
        ]

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        if self.error_calculator is not None:
            cer_ctc = []
            for i in range(self.num_spkrs):
                ys_hat = self.ctc.argmax(hs_pad[i].view(batch_size, -1, self.adim)).data
                cer_ctc.append(
                    self.error_calculator(ys_hat.cpu(), ys_pad[i].cpu(), is_ctc=True)
                )
            cer_ctc = sum(map(lambda x: x[0] * x[1], zip(cer_ctc, ys_out_len))) / sum(
                ys_out_len
            )
        else:
            cer_ctc = None

        # 3. forward decoder
        if self.mtlalpha == 1.0:
            loss_att, self.acc, cer, wer = None, None, None, None
        else:
            pred_pad, pred_mask = [None] * self.num_spkrs, [None] * self.num_spkrs
            loss_att, acc = [None] * self.num_spkrs, [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                (
                    pred_pad[i],
                    pred_mask[i],
                    loss_att[i],
                    acc[i],
                ) = self.decoder_and_attention(
                    hs_pad[i], hs_mask[i], ys_pad[i], batch_size
                )

            # 4. compute attention loss
            # The following is just an approximation
            loss_att = sum(map(lambda x: x[0] * x[1], zip(loss_att, ys_out_len))) / sum(
                ys_out_len
            )
            self.acc = sum(map(lambda x: x[0] * x[1], zip(acc, ys_out_len))) / sum(
                ys_out_len
            )

            # 5. compute cer/wer
            if self.training or self.error_calculator is None:
                cer, wer = None, None
            else:
                ys_hat = pred_pad.argmax(dim=-1)
                cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def decoder_and_attention(self, hs_pad, hs_mask, ys_pad, batch_size):
        """Forward decoder and attention loss."""
        # forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        return pred_pad, pred_mask, loss_att, acc

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output

    def recog(self, enc_output, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech of each speaker.

        :param ndnarray enc_output: encoder outputs (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection

            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recog(enc_output, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech of each speaker.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # Encoder
        enc_output = self.encode(x)

        # Decoder
        nbest_hyps = []
        for enc_out in enc_output:
            nbest_hyps.append(
                self.recog(enc_out, recog_args, char_list, rnnlm, use_jit)
            )
        return nbest_hyps
