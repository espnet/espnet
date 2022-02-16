# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer text translation model (pytorch)."""

from argparse import Namespace
import logging
import math

import numpy as np
import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_mt_common import ErrorCalculator
from espnet.nets.mt_interface import MTInterface
from espnet.nets.pytorch_backend.e2e_mt import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args


class E2E(MTInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")
        group = add_arguments_transformer_common(group)
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer="embed",
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.decoder = Decoder(
            odim=odim,
            selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_decoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.pad = 0  # use <blank> for padding
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="mt", arch="transformer")
        self.reporter = Reporter()

        # tie source and target emeddings
        if args.tie_src_tgt_embedding:
            if idim != odim:
                raise ValueError(
                    "When using tie_src_tgt_embedding, idim and odim must be equal."
                )
            self.encoder.embed[0].weight = self.decoder.embed[0].weight

        # tie emeddings and the classfier
        if args.tie_classifier:
            self.decoder.output_layer.weight = self.decoder.embed[0].weight

        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )
        self.normalize_length = args.transformer_length_normalized_loss  # for PPL
        self.reset_parameters(args)
        self.adim = args.adim
        self.error_calculator = ErrorCalculator(
            args.char_list, args.sym_space, args.sym_blank, args.report_bleu
        )
        self.rnnlm = None

        # multilingual MT related
        self.multilingual = args.multilingual

    def reset_parameters(self, args):
        """Initialize parameters."""
        initialize(self, args.transformer_init)
        torch.nn.init.normal_(
            self.encoder.embed[0].weight, mean=0, std=args.adim**-0.5
        )
        torch.nn.init.constant_(self.encoder.embed[0].weight[self.pad], 0)
        torch.nn.init.normal_(
            self.decoder.embed[0].weight, mean=0, std=args.adim**-0.5
        )
        torch.nn.init.constant_(self.decoder.embed[0].weight[self.pad], 0)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        xs_pad, ys_pad = self.target_forcing(xs_pad, ys_pad)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # 3. compute attention loss
        self.loss = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # 4. compute corpus-level bleu in a mini-batch
        if self.training:
            self.bleu = None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            self.bleu = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        loss_data = float(self.loss)
        if self.normalize_length:
            self.ppl = np.exp(loss_data)
        else:
            batch_size = ys_out_pad.size(0)
            ys_out_pad = ys_out_pad.view(-1)
            ignore = ys_out_pad == self.ignore_id  # (B*T,)
            total_n_tokens = len(ys_out_pad) - ignore.sum().item()
            self.ppl = np.exp(loss_data * batch_size / total_n_tokens)
        if not math.isnan(loss_data):
            self.reporter.report(loss_data, self.acc, self.ppl, self.bleu)
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, xs):
        """Encode source sentences."""
        self.eval()
        xs = torch.as_tensor(xs).unsqueeze(0)
        enc_output, _ = self.encoder(xs, None)
        return enc_output.squeeze(0)

    def target_forcing(self, xs_pad, ys_pad=None, tgt_lang=None):
        """Prepend target language IDs to source sentences for multilingual MT.

        These tags are prepended in source/target sentences as pre-processing.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :return: source text without language IDs
        :rtype: torch.Tensor
        :return: target text without language IDs
        :rtype: torch.Tensor
        :return: target language IDs
        :rtype: torch.Tensor (B, 1)
        """
        if self.multilingual:
            xs_pad = xs_pad[:, 1:]  # remove source language IDs here
            if ys_pad is not None:
                # remove language ID in the beginning
                lang_ids = ys_pad[:, 0].unsqueeze(1)
                ys_pad = ys_pad[:, 1:]
            elif tgt_lang is not None:
                lang_ids = xs_pad.new_zeros(xs_pad.size(0), 1).fill_(tgt_lang)
            else:
                raise ValueError("Set ys_pad or tgt_lang.")

            # prepend target language ID to source sentences
            xs_pad = torch.cat([lang_ids, xs_pad], dim=1)
        return xs_pad, ys_pad

    def translate(self, x, trans_args, char_list=None):
        """Translate source text.

        :param list x: input source text feature (T,)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        """
        self.eval()  # NOTE: this is important because self.encode() is not used
        assert isinstance(x, list)

        # make a utt list (1) to use the same interface for encoder
        if self.multilingual:
            x = to_device(
                self, torch.from_numpy(np.fromiter(map(int, x[0][1:]), dtype=np.int64))
            )
        else:
            x = to_device(
                self, torch.from_numpy(np.fromiter(map(int, x[0]), dtype=np.int64))
            )

        logging.info("input lengths: " + str(x.size(0)))
        xs_pad = x.unsqueeze(0)
        tgt_lang = None
        if trans_args.tgt_lang:
            tgt_lang = char_list.index(trans_args.tgt_lang)
        xs_pad, _ = self.target_forcing(xs_pad, tgt_lang=tgt_lang)
        h, _ = self.encoder(xs_pad, None)
        logging.info("encoder output lengths: " + str(h.size(1)))

        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        if trans_args.maxlenratio == 0:
            maxlen = h.size(1)
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * h.size(1)))
        minlen = int(trans_args.minlenratio * h.size(1))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        hyp = {"score": 0.0, "yseq": [self.sos]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            logging.debug("position " + str(i))

            # batchfy
            ys = h.new_zeros((len(hyps), i + 1), dtype=torch.int64)
            for j, hyp in enumerate(hyps):
                ys[j, :] = torch.tensor(hyp["yseq"])
            ys_mask = subsequent_mask(i + 1).unsqueeze(0).to(h.device)

            local_scores = self.decoder.forward_one_step(
                ys, ys_mask, h.repeat([len(hyps), 1, 1])
            )[0]

            hyps_best_kept = []
            for j, hyp in enumerate(hyps):
                local_best_scores, local_best_ids = torch.topk(
                    local_scores[j : j + 1], beam, dim=1
                )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
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
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
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
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, trans_args, char_list)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention) and m.attn is not None:
                ret[name] = m.attn.cpu().numpy()
        self.train()
        return ret
