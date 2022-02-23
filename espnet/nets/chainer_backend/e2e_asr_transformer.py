# encoding: utf-8
"""Transformer-based model for End-to-end ASR."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math

import chainer
import chainer.functions as F
from chainer import reporter
import numpy as np
import six

from espnet.nets.chainer_backend.asr_interface import ChainerASRInterface
from espnet.nets.chainer_backend.transformer.attention import MultiHeadAttention
from espnet.nets.chainer_backend.transformer import ctc
from espnet.nets.chainer_backend.transformer.decoder import Decoder
from espnet.nets.chainer_backend.transformer.encoder import Encoder
from espnet.nets.chainer_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.chainer_backend.transformer.training import CustomConverter
from espnet.nets.chainer_backend.transformer.training import CustomUpdater
from espnet.nets.chainer_backend.transformer.training import (
    CustomParallelUpdater,  # noqa: H301
)
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport


CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


class E2E(ChainerASRInterface):
    """E2E module.

    Args:
        idim (int): Input dimmensions.
        odim (int): Output dimmensions.
        args (Namespace): Training config.
        ignore_id (int, optional): Id for ignoring a character.
        flag_return (bool, optional): If true, return a list with (loss,
        loss_ctc, loss_att, acc) in forward. Otherwise, return loss.

    """

    @staticmethod
    def add_arguments(parser):
        """Customize flags for transformer setup.

        Args:
            parser (Namespace): Training config.

        """
        group = parser.add_argument_group("transformer model setting")
        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv2d", "linear", "embed"],
            help="transformer input layer type",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate",
            default=None,
            type=float,
            help="dropout in transformer attention. use --dropout-rate if None is set",
        )
        group.add_argument(
            "--transformer-lr",
            default=10.0,
            type=float,
            help="Initial value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-length-normalized-loss",
            default=True,
            type=strtobool,
            help="normalize loss by length",
        )

        group.add_argument(
            "--dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate for the encoder",
        )
        # Encoder
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        # Attention
        group.add_argument(
            "--adim",
            default=320,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        # Decoder
        group.add_argument(
            "--dlayers", default=1, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=320, type=int, help="Number of decoder hidden units"
        )
        return parser

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(np.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1, flag_return=True):
        """Initialize the transformer."""
        chainer.Chain.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0 <= self.mtlalpha <= 1, "mtlalpha must be [0,1]"
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.use_label_smoothing = False
        self.char_list = args.char_list
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.scale_emb = args.adim**0.5
        self.sos = odim - 1
        self.eos = odim - 1
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.ignore_id = ignore_id
        self.reset_parameters(args)
        with self.init_scope():
            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                initialW=self.initialW,
                initial_bias=self.initialB,
            )
            self.decoder = Decoder(
                odim, args, initialW=self.initialW, initial_bias=self.initialB
            )
            self.criterion = LabelSmoothingLoss(
                args.lsm_weight,
                len(args.char_list),
                args.transformer_length_normalized_loss,
            )
            if args.mtlalpha > 0.0:
                if args.ctc_type == "builtin":
                    logging.info("Using chainer CTC implementation")
                    self.ctc = ctc.CTC(odim, args.adim, args.dropout_rate)
                elif args.ctc_type == "warpctc":
                    logging.info("Using warpctc CTC implementation")
                    self.ctc = ctc.WarpCTC(odim, args.adim, args.dropout_rate)
                else:
                    raise ValueError(
                        'ctc_type must be "builtin" or "warpctc": {}'.format(
                            args.ctc_type
                        )
                    )
            else:
                self.ctc = None
        self.dims = args.adim
        self.odim = odim
        self.flag_return = flag_return
        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        if "Namespace" in str(type(args)):
            self.verbose = 0 if "verbose" not in args else args.verbose
        else:
            self.verbose = 0 if args.verbose is None else args.verbose

    def reset_parameters(self, args):
        """Initialize the Weight according to the give initialize-type.

        Args:
            args (Namespace): Transformer config.

        """
        type_init = args.transformer_init
        if type_init == "lecun_uniform":
            logging.info("Using LeCunUniform as Parameter initializer")
            self.initialW = chainer.initializers.LeCunUniform
        elif type_init == "lecun_normal":
            logging.info("Using LeCunNormal as Parameter initializer")
            self.initialW = chainer.initializers.LeCunNormal
        elif type_init == "gorot_uniform":
            logging.info("Using GlorotUniform as Parameter initializer")
            self.initialW = chainer.initializers.GlorotUniform
        elif type_init == "gorot_normal":
            logging.info("Using GlorotNormal as Parameter initializer")
            self.initialW = chainer.initializers.GlorotNormal
        elif type_init == "he_uniform":
            logging.info("Using HeUniform as Parameter initializer")
            self.initialW = chainer.initializers.HeUniform
        elif type_init == "he_normal":
            logging.info("Using HeNormal as Parameter initializer")
            self.initialW = chainer.initializers.HeNormal
        elif type_init == "pytorch":
            logging.info("Using Pytorch initializer")
            self.initialW = chainer.initializers.Uniform
        else:
            logging.info("Using Chainer default as Parameter initializer")
            self.initialW = chainer.initializers.Uniform
        self.initialB = chainer.initializers.Uniform

    def forward(self, xs, ilens, ys_pad, calculate_attentions=False):
        """E2E forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded character ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)
            ys (chainer.Variable): Batch of padded target features. (B, Lmax, odim)
            calculate_attentions (bool): If true, return value is the output of encoder.

        Returns:
            float: Training loss.
            float (optional): Training loss for ctc.
            float (optional): Training loss for attention.
            float (optional): Accuracy.
            chainer.Variable (Optional): Output of the encoder.

        """
        alpha = self.mtlalpha

        # 1. Encoder
        xs, x_mask, ilens = self.encoder(xs, ilens)

        # 2. CTC loss
        cer_ctc = None
        if alpha == 0.0:
            loss_ctc = None
        else:
            _ys = [y.astype(np.int32) for y in ys_pad]
            loss_ctc = self.ctc(xs, _ys)
            if self.error_calculator is not None:
                with chainer.no_backprop_mode():
                    ys_hat = chainer.backends.cuda.to_cpu(self.ctc.argmax(xs).data)
                cer_ctc = self.error_calculator(ys_hat, ys_pad, is_ctc=True)

        # 3. Decoder
        if calculate_attentions:
            self.calculate_attentions(xs, x_mask, ys_pad)
        ys = self.decoder(ys_pad, xs, x_mask)

        # 4. Attention Loss
        cer, wer = None, None
        if alpha == 1:
            loss_att = None
            acc = None
        else:
            # Make target
            eos = np.array([self.eos], "i")
            with chainer.no_backprop_mode():
                ys_pad_out = [np.concatenate([y, eos], axis=0) for y in ys_pad]
                ys_pad_out = F.pad_sequence(ys_pad_out, padding=-1).data
                ys_pad_out = self.xp.array(ys_pad_out)

            loss_att = self.criterion(ys, ys_pad_out)
            acc = F.accuracy(
                ys.reshape(-1, self.odim), ys_pad_out.reshape(-1), ignore_label=-1
            )
            if (not chainer.config.train) and (self.error_calculator is not None):
                cer, wer = self.error_calculator(ys, ys_pad)

        if alpha == 0.0:
            self.loss = loss_att
            loss_att_data = loss_att.data
            loss_ctc_data = None
        elif alpha == 1.0:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = loss_ctc.data
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = loss_att.data
            loss_ctc_data = loss_ctc.data
        loss_data = self.loss.data

        if not math.isnan(loss_data):
            reporter.report({"loss_ctc": loss_ctc_data}, self)
            reporter.report({"loss_att": loss_att_data}, self)
            reporter.report({"acc": acc}, self)

            reporter.report({"cer_ctc": cer_ctc}, self)
            reporter.report({"cer": cer}, self)
            reporter.report({"wer": wer}, self)

            logging.info("mtl loss:" + str(loss_data))
            reporter.report({"loss": loss_data}, self)
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        if self.flag_return:
            loss_ctc = None
            return self.loss, loss_ctc, loss_att, acc
        else:
            return self.loss

    def calculate_attentions(self, xs, x_mask, ys_pad):
        """Calculate Attentions."""
        self.decoder(ys_pad, xs, x_mask)

    def recognize(self, x_block, recog_args, char_list=None, rnnlm=None):
        """E2E recognition function.

        Args:
            x (ndarray): Input acouctic feature (B, T, D) or (T, D).
            recog_args (Namespace): Argment namespace contraining options.
            char_list (List[str]): List of characters.
            rnnlm (chainer.Chain): Language model module defined at
            `espnet.lm.chainer_backend.lm`.

        Returns:
            List: N-best decoding results.

        """
        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            # 1. encoder
            ilens = [x_block.shape[0]]
            batch = len(ilens)
            xs, _, _ = self.encoder(x_block[None, :, :], ilens)

            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(xs.reshape(batch, -1, self.dims)).data[0]
            else:
                lpz = None
            # 2. decoder
            if recog_args.lm_weight == 0.0:
                rnnlm = None
            y = self.recognize_beam(xs, lpz, recog_args, char_list, rnnlm)

        return y

    def recognize_beam(self, h, lpz, recog_args, char_list=None, rnnlm=None):
        """E2E beam search.

        Args:
            h (ndarray): Encoder output features (B, T, D) or (T, D).
            lpz (ndarray): Log probabilities from CTC.
            recog_args (Namespace): Argment namespace contraining options.
            char_list (List[str]): List of characters.
            rnnlm (chainer.Chain): Language model module defined at
            `espnet.lm.chainer_backend.lm`.

        Returns:
            List: N-best decoding results.

        """
        logging.info("input lengths: " + str(h.shape[1]))

        # initialization
        n_len = h.shape[1]
        xp = self.xp
        h_mask = xp.ones((1, n_len))

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # prepare sos
        y = self.sos
        if recog_args.maxlenratio == 0:
            maxlen = n_len
        else:
            maxlen = max(1, int(recog_args.maxlenratio * n_len))
        minlen = int(recog_args.minlenratio * n_len)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}

        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz, 0, self.eos, self.xp)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]

        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                ys = F.expand_dims(xp.array(hyp["yseq"]), axis=0).data
                out = self.decoder(ys, h, h_mask)

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(out[:, -1], axis=-1).data
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(
                        hyp["rnnlm_prev"], hyp["yseq"][i]
                    )
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_ids = xp.argsort(local_scores, axis=1)[0, ::-1][
                        :ctc_beam
                    ]
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids, hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids
                    ] + ctc_weight * (ctc_scores - hyp["ctc_score_prev"])
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids]
                        )
                    joint_best_ids = xp.argsort(local_scores, axis=1)[0, ::-1][:beam]
                    local_best_scores = local_scores[:, joint_best_ids]
                    local_best_ids = local_best_ids[joint_best_ids]
                else:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][
                        :beam
                    ]
                    local_best_scores = local_scores[:, local_best_ids]

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[j]]
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothesis: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                    + " score: "
                    + str(hyps[0]["score"])
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
                logging.debug("remained hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break
            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x["score"], reverse=True
        )  # [:min(len(ended_hyps), recog_args.nbest)]

        logging.debug(nbest_hyps)
        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        # remove sos
        return nbest_hyps

    def calculate_all_attentions(self, xs, ilens, ys):
        """E2E attention calculation.

        Args:
            xs (List[tuple()]): List of padded input sequences.
                [(T1, idim), (T2, idim), ...]
            ilens (ndarray): Batch of lengths of input sequences. (B)
            ys (List): List of character id sequence tensor. [(L1), (L2), (L3), ...]

        Returns:
            float ndarray: Attention weights. (B, Lmax, Tmax)

        """
        with chainer.no_backprop_mode():
            self(xs, ilens, ys, calculate_attentions=True)
        ret = dict()
        for name, m in self.namedlinks():
            if isinstance(m, MultiHeadAttention):
                var = m.attn
                var.to_cpu()
                _name = name[1:].replace("/", "_")
                ret[_name] = var.data
        return ret

    @property
    def attention_plot_class(self):
        """Attention plot function.

        Redirects to PlotAttentionReport

        Returns:
            PlotAttentionReport

        """
        return PlotAttentionReport

    @staticmethod
    def custom_converter(subsampling_factor=0):
        """Get customconverter of the model."""
        return CustomConverter()

    @staticmethod
    def custom_updater(iters, optimizer, converter, device=-1, accum_grad=1):
        """Get custom_updater of the model."""
        return CustomUpdater(
            iters, optimizer, converter=converter, device=device, accum_grad=accum_grad
        )

    @staticmethod
    def custom_parallel_updater(iters, optimizer, converter, devices, accum_grad=1):
        """Get custom_parallel_updater of the model."""
        return CustomParallelUpdater(
            iters,
            optimizer,
            converter=converter,
            devices=devices,
            accum_grad=accum_grad,
        )
