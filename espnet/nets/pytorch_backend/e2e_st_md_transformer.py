# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math
import numpy
import sys

import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_st import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_md_transformer_common,  # noqa: H301
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
from espnet.nets.st_interface import STInterface
from espnet.utils.fill_missing_args import fill_missing_args


class E2E(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")
        group = add_arguments_md_transformer_common(group)
        # in order to reuse most of the arguments
        # we call encoder as the one in the input side (ie speech encoder)
        # we call decoder as the one in the output side (ie translation decoder)
        # we call the intermediate decoder as the input side (ie asr decoder)

        # initialization related
        group.add_argument(
            "--init-like-bert-enc",
            default=False,
            type=strtobool,
            help="Initialize decoder parameters like BERT",
        )
        group.add_argument(
            "--init-like-bert-dec",
            default=False,
            type=strtobool,
            help="Initialize decoder parameters like BERT",
        )
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder_asr.conv_subsampling_factor * int(
            numpy.prod(self.subsample)
        )

    def __init__(self, idim, odim, args, ignore_id=-1, odim_si=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        # eunits == enc_inp_units
        # elayers == enc_inp_layers
        # adim == enc_inp_adim
        # aheads == enc_inp_aheads
        # transformer_attn_dropout_rate == enc_inp_transformer_attn_dropout_rate
        # dlayers == dec_out_layers
        # dunits == dec_out_units
        # transformer_decoder_selfattn_layer_type == \
        #   dec_si_transformer_selfattn_layer_type
        # enc_inp_input_layer = transformer_input_layer,
        if args.enc_inp_dropout_rate is None:
            args.enc_inp_dropout_rate = args.dropout_rate
        if args.enc_inp_transformer_attn_dropout_rate is None:
            args.enc_inp_transformer_attn_dropout_rate = args.dropout_rate

        if args.dec_out_aheads is None:
            args.dec_out_aheads = args.enc_inp_aheads
        if args.dec_out_adim is None:
            args.dec_out_adim = args.enc_inp_adim
        if args.dec_out_dropout_rate is None:
            args.dec_out_dropout_rate = args.dropout_rate
        if args.dec_out_transformer_attn_dropout_rate is None:
            args.dec_out_transformer_attn_dropout_rate = (
                args.enc_inp_transformer_attn_dropout_rate
            )
        if args.dec_out_transformer_selfattn_layer_type is None:
            args.dec_out_transformer_selfattn_layer_type = (
                args.dec_si_transformer_selfattn_layer_type
            )

        if args.dec_si_units is None:
            args.dec_si_units = args.dec_out_units
        if args.dec_si_aheads is None:
            args.dec_si_aheads = args.enc_inp_aheads
        if args.dec_si_adim is None:
            args.dec_si_adim = args.enc_inp_adim
        if args.dec_si_dropout_rate is None:
            args.dec_si_dropout_rate = args.dec_out_dropout_rate
        if args.dec_si_transformer_attn_dropout_rate is None:
            args.dec_si_transformer_attn_dropout_rate = (
                args.dec_out_transformer_attn_dropout_rate
            )

        if args.enc_si_units is None:
            args.enc_si_units = args.enc_inp_units
        if args.enc_si_aheads is None:
            args.enc_si_aheads = args.enc_inp_aheads
        if args.enc_si_adim is None:
            args.enc_si_adim = args.enc_inp_adim
        if args.enc_si_dropout_rate is None:
            args.enc_si_dropout_rate = args.enc_inp_dropout_rate
        if args.enc_si_transformer_attn_dropout_rate is None:
            args.enc_si_transformer_attn_dropout_rate = (
                args.enc_inp_transformer_attn_dropout_rate
            )
        if args.enc_si_transformer_selfattn_layer_type is None:
            args.enc_si_transformer_selfattn_layer_type = (
                args.transformer_encoder_selfattn_layer_type
            )

        self.pad = 0  # use <blank> for padding
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="st", arch="transformer")
        self.reporter = Reporter()

        if odim_si == -1:
            self.odim_si = self.odim
            self.sos_si = self.sos
            self.eos_si = self.eos
        else:
            self.odim_si = odim_si
            self.sos_si = odim_si - 1
            self.eos_si = odim_si - 1

        if "speechattn" in args.dec_out_transformer_selfattn_layer_type:
            logging.warning("Using speech attention")
            self.use_speech_attn = True
        else:
            self.use_speech_attn = False

        self.encoder_asr = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.enc_inp_adim,
            attention_heads=args.enc_inp_aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.enc_inp_units,
            num_blocks=args.enc_inp_layers,
            input_layer=args.enc_inp_input_layer,
            dropout_rate=args.enc_inp_dropout_rate,
            positional_dropout_rate=args.enc_inp_dropout_rate,
            attention_dropout_rate=args.enc_inp_transformer_attn_dropout_rate,
        )

        self.decoder_asr = Decoder(
            odim=self.odim_si,
            selfattention_layer_type=args.dec_si_transformer_selfattn_layer_type,
            attention_dim=args.dec_si_adim,
            attention_heads=args.dec_si_aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_decoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.dec_si_units,
            num_blocks=args.dec_si_layers,
            dropout_rate=args.dec_si_dropout_rate,
            positional_dropout_rate=args.dec_si_dropout_rate,
            self_attention_dropout_rate=args.dec_si_transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.dec_si_transformer_attn_dropout_rate,
        )

        # FIXXX
        if args.enc_si_layers > 0:
            self.encoder_st = Encoder(
                idim=self.odim_si,
                selfattention_layer_type=args.enc_si_transformer_selfattn_layer_type,
                attention_dim=args.enc_si_adim,
                attention_heads=args.enc_si_aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_encoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.enc_si_units,
                num_blocks=args.enc_si_layers,
                input_layer=args.enc_si_input_layer,
                dropout_rate=args.enc_si_dropout_rate,
                positional_dropout_rate=args.enc_si_dropout_rate,
                attention_dropout_rate=args.enc_si_transformer_attn_dropout_rate,
            )

        self.decoder_st = Decoder(
            odim=odim,
            selfattention_layer_type=args.dec_out_transformer_selfattn_layer_type,
            attention_dim=args.dec_out_adim,
            attention_heads=args.dec_out_aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_decoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.dec_out_units,
            num_blocks=args.dec_out_layers,
            dropout_rate=args.dec_out_dropout_rate,
            positional_dropout_rate=args.dec_out_dropout_rate,
            self_attention_dropout_rate=args.dec_out_transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.dec_out_transformer_attn_dropout_rate,
        )

        self.criterion_st = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.criterion_asr = LabelSmoothingLoss(
            self.odim_si,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        # submodule for ASR task
        self.mtlalpha = args.mtlalpha
        self.asr_weight = args.asr_weight

        if self.asr_weight == 0.0:
            logging.warning("asr_weghts can't be 0: Needs ASR searchable intermediates")
            sys.exit(1)

        self.reset_parameters(args)  # NOTE: place after the submodule initialization

        self.enc_inp_adim = args.enc_inp_adim  # used for CTC (equal to d_model)
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                self.odim_si,
                self.enc_inp_adim,
                args.dropout_rate,
                ctc_type=args.ctc_type,
                reduce=True,
            )
        else:
            self.ctc = None

        # translation error calculator
        self.error_calculator = MTErrorCalculator(
            args.char_list, args.sym_space, args.sym_blank, args.report_bleu
        )

        # recognition error calculator
        self.error_calculator_asr = ASRErrorCalculator(
            args.char_list,
            args.sym_space,
            args.sym_blank,
            args.report_cer,
            args.report_wer,
        )

        self.rnnlm = None

        # multilingual E2E-ST related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = getattr(args, "replace_sos", False)

        if args.init_like_bert_enc:
            self.init_like_bert_enc()
        if args.init_like_bert_dec:
            self.init_like_bert_dec()

    def reset_parameters(self, args):
        """Initialize parameters."""
        initialize(self, args.transformer_init)

    def _init_like_bert(self, n, p):
        if "embed" in n:
            return
        if "norm" in n and "weight" in n:
            assert p.dim() == 1
            torch.nn.init.normal_(p, mean=1.0, std=0.02)  # gamma in layer normalization
        elif p.dim() == 1:
            torch.nn.init.constant_(p, 0.0)  # bias
        elif p.dim() == 2:
            torch.nn.init.normal_(p, mean=0, std=0.02)

    def init_like_bert_enc(self):
        """Initialize like bert."""
        # ASR encoder
        for n, p in self.encoder_asr.named_parameters():
            self._init_like_bert(n, p)
        # ST encoder
        for n, p in self.encoder_st.named_parameters():
            self._init_like_bert(n, p)

    def init_like_bert_dec(self):
        """Initialize like bert."""
        # ASR decoder
        # for n, p in self.decoder_asr.named_parameters():
        #     self._init_like_bert(n, p)
        # ST decoder
        for n, p in self.decoder_st.named_parameters():
            self._init_like_bert(n, p)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Extract target language ID
        tgt_lang_ids = None
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining

        # 1. forward encoder_asr
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        speech_hs, speech_mask = self.encoder_asr(xs_pad, src_mask)

        # 2. Run ASR decoder and get last but one hidden representation
        (
            loss_asr_att,
            acc_asr,
            loss_asr_ctc,
            cer_ctc,
            cer,
            wer,
            hs_dec_asr,
        ) = self.forward_asr(speech_hs, speech_mask, ys_pad_src, get_hidden=True)
        ilens_asr = torch.sum(ys_pad_src != self.ignore_id, dim=1).cpu().numpy()
        ilens_asr = ilens_asr + 1

        src_mask_asr = (
            (make_non_pad_mask(ilens_asr.tolist())).to(hs_dec_asr.device).unsqueeze(-2)
        )

        # 3. forward intermediates encoder-st
        if hasattr(self, "encoder_st"):
            hs_pad, hs_mask = self.encoder_st(hs_dec_asr, src_mask_asr)
        else:
            # just pass the ASR decoder hs directly to the ST decoder
            hs_pad, hs_mask = hs_dec_asr, src_mask_asr

        # 2. forward decoder ST
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # replace <sos> with target language ID
        if self.replace_sos:
            ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        if self.use_speech_attn:
            pred_pad, pred_mask = self.decoder_st(
                ys_in_pad, ys_mask, hs_pad, hs_mask, speech_hs, speech_mask
            )
        else:
            pred_pad, pred_mask = self.decoder_st(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # 3. compute ST loss
        loss_att = self.criterion_st(pred_pad, ys_out_pad)

        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # 4. compute corpus-level bleu in a mini-batch
        if self.training:
            self.bleu = None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            self.bleu = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # 6. compute auxiliary MT loss

        asr_ctc_weight = self.mtlalpha
        self.loss = (1 - self.asr_weight) * loss_att + self.asr_weight * (
            asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
        )

        loss_asr_data = float(
            asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
        )

        acc_mt = None
        loss_mt_data = None
        loss_st_data = float(loss_att)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_asr_data,
                loss_mt_data,
                loss_st_data,
                acc_asr,
                acc_mt,
                self.acc,
                cer_ctc,
                cer,
                wer,
                self.bleu,
                loss_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def forward_asr(self, hs_pad, hs_mask, ys_pad, get_hidden=True):
        """Forward pass in the auxiliary ASR task.

        :param torch.Tensor hs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hs_mask: batch of input token mask (B, Lmax)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ASR attention loss value
        :rtype: torch.Tensor
        :return: accuracy in ASR attention decoder
        :rtype: float
        :return: ASR CTC loss value
        :rtype: torch.Tensor
        :return: character error rate from CTC prediction
        :rtype: float
        :return: character error rate from attetion decoder prediction
        :rtype: float
        :return: word error rate from attetion decoder prediction
        :rtype: float
        """
        loss_att, loss_ctc = 0.0, 0.0
        acc = None
        cer, wer = None, None
        cer_ctc = None
        if self.asr_weight == 0:
            return loss_att, acc, loss_ctc, cer_ctc, cer, wer

        # attention
        if self.mtlalpha < 1:
            ys_in_pad_asr, ys_out_pad_asr = add_sos_eos(
                ys_pad, self.sos_si, self.eos_si, self.ignore_id
            )
            ys_mask_asr = target_mask(ys_in_pad_asr, self.ignore_id)
            pred_pad, tgt_mask, hs_dec_asr = self.decoder_asr(
                ys_in_pad_asr, ys_mask_asr, hs_pad, hs_mask, return_hidden=get_hidden
            )
            loss_att = self.criterion_asr(pred_pad, ys_out_pad_asr)

            acc = th_accuracy(
                pred_pad.view(-1, self.odim_si),
                ys_out_pad_asr,
                ignore_label=self.ignore_id,
            )
            if not self.training:
                ys_hat_asr = pred_pad.argmax(dim=-1)
                cer, wer = self.error_calculator_asr(ys_hat_asr.cpu(), ys_pad.cpu())

        # CTC
        if self.mtlalpha > 0:
            batch_size = hs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(
                hs_pad.view(batch_size, -1, self.enc_inp_adim), hs_len, ys_pad
            )
            if not self.training:
                ys_hat_ctc = self.ctc.argmax(
                    hs_pad.view(batch_size, -1, self.enc_inp_adim)
                ).data
                cer_ctc = self.error_calculator_asr(
                    ys_hat_ctc.cpu(), ys_pad.cpu(), is_ctc=True
                )
                # for visualization
                self.ctc.softmax(hs_pad)

        return loss_att, acc, loss_ctc, cer_ctc, cer, wer, hs_dec_asr

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder_st)

    def encode_asr(self, x, mask=False):
        """Encode source acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        if mask:
            ilens = [len(x[0])]
            enc_mask = (make_non_pad_mask(ilens)).to(x.device).unsqueeze(-2)
            enc_output, enc_mask = self.encoder_asr(x, enc_mask)
            return enc_output, enc_mask

        enc_output, _ = self.encoder_asr(x, None)
        return enc_output.squeeze(0)

    def encode_st(self, x):
        """Encode ASR decoder intermediates."""
        self.eval()
        x = x.unsqueeze(0)
        enc_output, _ = self.encoder_st(x, None)
        return enc_output.squeeze(0)

    def encode(self, x, trans_args, char_list=None):
        """ASR encoder then ASR decoder then ST encoder."""
        # interface for ensemble
        x = torch.as_tensor(x)

        if trans_args.eval_st_subnet:
            asr_output, speech_enc = self.recognize(
                x, trans_args, char_list, None, None
            )
            x = asr_output.squeeze(0)
            maxlen_asr = len(speech_enc[0])
        else:
            asr_output, maxlen_asr, speech_enc = self.recognize(
                x, trans_args, char_list, None, None
            )
            x = torch.stack(asr_output[0]["hs_asrs"]).squeeze(1)

        if hasattr(self, "encoder_st"):
            st_enc_output = self.encode_st(x)
        else:
            st_enc_output = None
        return st_enc_output, speech_enc

    def decoder_forward_one_step(self, h, i, hyps):
        """Decode single step."""
        enc_output = h[0]
        speech_enc = h[1]
        stack_scores = []
        for hyp in hyps:

            # get nbest local scores and their ids
            ys_mask = subsequent_mask(i + 1).unsqueeze(0)
            ys = torch.tensor(hyp["yseq"]).unsqueeze(0)

            if self.use_speech_attn:
                local_att_scores = self.decoder_st.forward_one_step(
                    ys, ys_mask, enc_output, speech_enc
                )[0]
            else:
                local_att_scores = self.decoder_st.forward_one_step(
                    ys, ys_mask, enc_output
                )[0]

            # TODO(jiatong): skip lm
            stack_scores.append(local_att_scores.squeeze(0))
        return torch.stack(stack_scores, dim=0)

    def translate(
        self,
        x,
        trans_args,
        char_list=None,
        rnnlm=None,
        asr_target=None,
        asr_rnnlm=None,
    ):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos
        logging.info("Decoder ST: <sos> index: " + str(y))
        logging.info("Decoder ST: <sos> mark: " + char_list[y])
        logging.info("Decoder ST: input lengths: " + str(x.shape[0]))

        if trans_args.eval_st_subnet:
            asr_output, speech_enc = self.recognize(
                x, trans_args, char_list, asr_rnnlm, asr_target
            )
            x = asr_output.squeeze(0)
            maxlen_asr = len(speech_enc[0])
        else:
            asr_output, maxlen_asr, speech_enc = self.recognize(
                x, trans_args, char_list, asr_rnnlm, asr_target
            )
            x = torch.stack(asr_output[0]["hs_asrs"]).squeeze(1)

        if hasattr(self, "encoder_st"):
            enc_output = self.encode_st(x).unsqueeze(0)
            h = enc_output.squeeze(0)
        else:
            # skip second encoder
            enc_output = x.unsqueeze(0)
            h = x

        logging.info("Decoder 2: ASR output lengths: " + str(h.size(0)))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = maxlen_asr  # h.shape[0]
        else:
            # maxlen >= 1
            # maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
            maxlen = max(1, int(trans_args.maxlenratio * maxlen_asr))
        # minlen = int(trans_args.minlenratio * h.size(0))
        minlen = int(trans_args.minlenratio * maxlen_asr)
        logging.info("Decoder 2: max output length: " + str(maxlen))
        logging.info("Decoder 2: min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)

                if self.use_speech_attn:
                    local_att_scores = self.decoder_st.forward_one_step(
                        ys, ys_mask, enc_output, speech_enc
                    )[0]
                else:
                    local_att_scores = self.decoder_st.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + trans_args.lm_weight * local_lm_scores
                    )

                else:
                    local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1
                )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
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
                logging.info("Decoder 2: adding <eos> in the last postion in the loop")
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
                            hyp["score"] += trans_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("Decoder 2: end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("Decoder 2: no hypothesis. Finish decoding.")
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
            return self.translate(x, trans_args, char_list, rnnlm)

        logging.info("Decoder 2: total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "Decoder 2: normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )

        return nbest_hyps, asr_output

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, asr_target=None):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        if recog_args.eval_st_subnet:
            enc_output, enc_mask = self.encode_asr(x, mask=True)
            # 2. Run ASR decoder and get last but one hidden representation
            asr_target = torch.LongTensor(asr_target)
            _, _, _, _, _, _, hs_dec_asr = self.forward_asr(
                enc_output, enc_mask, asr_target, get_hidden=True
            )
            return hs_dec_asr, enc_output

        enc_output = self.encode_asr(x).unsqueeze(0)

        if self.mtlalpha > 0 and recog_args.asr_si_ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        if rnnlm is not None:
            logging.info("Using RNNLM")

        logging.info("Speech input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.asr_si_beam_size
        penalty = recog_args.asr_si_penalty
        ctc_weight = recog_args.asr_si_ctc_weight

        # preprare sos
        y = self.sos_si
        vy = h.new_zeros(1).long()

        if recog_args.asr_si_maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.asr_si_maxlenratio * h.size(0)))
        minlen = int(recog_args.asr_si_minlenratio * h.size(0))
        logging.info("Decoder 1 max output length: " + str(maxlen))
        logging.info("Decoder 1 min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None, "hs_asrs": []}
        else:
            hyp = {"score": 0.0, "yseq": [y], "hs_asrs": []}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(
                lpz.detach().numpy(), 0, self.eos_si, numpy
            )
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)

                local_att_scores, hs_asr = self.decoder_asr.forward_one_step(
                    ys,
                    ys_mask,
                    enc_output,
                    return_hidden=True,
                )[:2]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.asr_si_lm_weight * local_lm_scores
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
                            recog_args.asr_si_lm_weight
                            * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    new_hyp["hs_asrs"] = [] + hyp["hs_asrs"]
                    new_hyp["hs_asrs"].append(hs_asr)
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

            #
            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos_si:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.asr_si_lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.asr_si_maxlenratio == 0.0:
                logging.info("Decoder 1: end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("Decoder 1: no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.asr_si_nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return hyps, maxlen, enc_output

        logging.info("Decoder 1: total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "Decoder 1: normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps, maxlen, enc_output

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention) and m.attn is not None
            ):  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = None
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
