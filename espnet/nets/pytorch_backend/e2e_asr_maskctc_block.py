# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
from espnet.nets.pytorch_backend.transformer.overlapping_interface import (
    tie_breaking,
    dynamic_matching,
    dynamic_matching_xl,
    get_token_level_ids_probs,
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.maskctc.mask import square_mask
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.e2e_asr_maskctc import E2E as E2EMaskctc
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.transformer.encoder_xl import EncoderXL as Encoder
from chainer import reporter
import chainer
import torch
from numpy.random import uniform
import numpy
from distutils.util import strtobool
import pdb
import time
import math
import logging
from itertools import groupby
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Mask CTC based non-autoregressive speech recognition model (pytorch).

See https://arxiv.org/abs/2005.08700 for the detail.

"""


from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
)
# from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform2 as mask_uniform
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
        LabelSmoothingLoss,  # noqa: H301
    )

class E2E(E2ETransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_maskctc_arguments(parser)
        E2E.add_block_attention_arguments(parser)

        return parser

    @staticmethod
    def add_maskctc_arguments(parser):
        """Add arguments for maskctc model."""
        group = parser.add_argument_group("maskctc specific setting")

        group.add_argument(
            "--maskctc-use-conformer-encoder",
            default=False,
            type=strtobool,
        )
        group.add_argument(
            "--maskctc-dynamic-length-prediction-weight",
            default=0.0,
            type=float,
        )
        group = add_arguments_conformer_common(group)

        return parser

    @staticmethod
    def add_block_attention_arguments(parser):
        """Add arguments for maskctc model."""
        group = parser.add_argument_group("block attention specific setting")

        group.add_argument(
            "--block-length",
            default=32, type=int,
            help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        odim += 1
        logging.info("odim {}".format(odim))
        super().__init__(idim, odim, args, ignore_id)
        assert 0.0 <= self.mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"
        self.mask_token = odim - 1
        self.sos = odim - 2
        self.eos = odim - 2
        self.block_len = args.block_length
        # <mask token>

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            block_length=args.block_length,
        )

        self.reset_parameters(args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        if self.block_len == -1:
            alpha = uniform(0, 1)
            if alpha > 0.5:
                block_len = -1
            else:
                block_len = int(uniform(1, 64))
                logging.info("block length is {}".format(block_len))
        else:
            block_len = self.block_len

        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(
            xs_pad.device).unsqueeze(-2)

        hs_pad, hs_mask, _, _, bl = self.encoder(
            xs_pad, src_mask, None, None, bl=block_len)
        self.hs_pad = hs_pad

        # 2. forward decoder
        ys_in_pad, ys_out_pad = mask_uniform(
            ys_pad, self.mask_token, self.eos, self.ignore_id
        )
        ys_mask = square_mask(ys_in_pad, self.eos)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        # logging.info("ys_in_pad:{}".format(ys_in_pad[0:3]))
        # logging.info("pred_pad:{}".format(pred_pad[0:3].argmax(dim=-1)))
        # logging.info("ys_out_pad:{}".format(ys_out_pad[0:3]))
        self.pred_pad = pred_pad

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        logging.info("acc, loss_att:{}, {}".format(self.acc, loss_att))

        # 4. compute ctc loss
        loss_ctc, cer_ctc = None, None
        if self.mtlalpha > 0:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(
                batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(
                    batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(
                    ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
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

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: decoding result
        :rtype: list
        """
        def num2str(char_list, mask_token, mask_char="_"):
            def f(yl):
                cl = [char_list[y] if y != mask_token else mask_char for y in yl]
                return "".join(cl).replace("<space>", " ")

            return f

        n2s = num2str(char_list, self.mask_token)

        x = torch.as_tensor(x)
        self.eval()
        logging.info(
            "check model state mask token: {}".format(self.mask_token))
        block_len = int(recog_args.block_length)
        logging.info("block length:{}".format(block_len))
        bh = block_len//2
        chuck_len = 1
        subsample = 4
        cache_len = 3
        decode_mode = recog_args.streaming_mode
        logging.info("self.odim is {}".format(self.odim))
        if decode_mode == "window":
            x = x[:x.size(0)//subsample * subsample, :]
            h_pad = torch.zeros(x.size(0)//subsample,
                                self.adim, dtype=x.dtype).unsqueeze(0)

            logging.info(
                "length length//4 {} {}".format(x.size(0), x.size(0)//subsample))

            hyp_new = [self.sos]
            t0 = 0
            t1 = 0
            ret = []
            mask_cache = torch.tensor([], dtype=torch.bool)
            y_in_cache = torch.tensor([[]], dtype=torch.long)
            pred_score_cache = torch.tensor([], dtype=torch.float)
            y_in_length_cache = []
            ctc_ids_prev = torch.tensor([[]], dtype=torch.long)
            y_hat_prev = torch.tensor([], dtype=torch.float)
            dp, tensor1, tensor2 = None, None, None
            rt, rp = None, None
            for t in range(x.size(0)):
                # with streaming input, get the hidden output of encoder
                if (t+1) % (bh * subsample) == 0 or (t == x.size(0) - 1 and t1//subsample < (t+1)//subsample):

                    logging.info("chunking feat {} {} {}".format(
                        t1, t+1, x[t1:t+1, :].size()))
                    if t1 >= block_len * subsample * (chuck_len + 1/2):
                        # after first block
                        hs = self.encoder(x[t1-block_len*subsample*chuck_len - bh*subsample:t+1, :]
                                          .unsqueeze(0), None, None, None)[0][:, block_len*chuck_len:, :]
                    else:
                        hs = self.encoder(x[:t+1, :]
                                          .unsqueeze(0), None, None, None)[0][:, (t+1)//subsample-block_len:, ]
                    if t1 >= block_len * subsample // 2:
                        t0 = t1 - block_len * subsample // 2
                    else:
                        t0 = t1

                    # greedy ctc output
                    logging.info("intervel: {}, {}".format(
                        t1//subsample, t//subsample))

                    ctc_probs_curr, ctc_ids_curr = torch.exp(self.ctc.log_softmax(
                        hs)).max(dim=-1)
                    if ctc_ids_curr.size(1) % bh > 0:

                        res = ctc_ids_curr.size(1) % bh
                        ctc_probs_curr = torch.nn.functional.pad(
                            ctc_probs_curr, (0, bh-res))
                        ctc_ids_curr = torch.nn.functional.pad(
                            ctc_ids_curr, (0, bh-res))

                    # get the y_hat and token level probs
                    y_hat_curr, probs_hat_curr, cidx = get_token_level_ids_probs(
                        ctc_ids_curr, ctc_probs_curr)
                    # get pairs
                    if t1 >= block_len * subsample:
                        dp, pairs, probs, tensor1, tensor2, rt, rp = dynamic_matching_xl2(
                            y_hat_prev[cidx_prev:],
                            y_hat_curr[:cidx],
                            probs_hat_prev[cidx_prev:],
                            probs_hat_curr[:cidx],
                            None, tensor1, tensor2,
                            rt, rp)

                        y_hat, probs_hat = tie_breaking(pairs, probs)
                    else:
                        y_hat = y_hat_curr[:cidx, 0]
                        probs_hat = probs_hat_curr[:cidx]
                    # assign prev y_hat
                    if t1 >= block_len * subsample // 2:
                        y_hat_prev = y_hat_curr.clone().detach()
                        probs_hat_prev = probs_hat_curr.clone(
                        ).detach()
                        cidx_prev = cidx
                        ctc_ids_prev = ctc_ids_curr.clone().detach()
                    y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

                    # mask ctc output based on ctc probablities
                    p_thres = recog_args.maskctc_probability_threshold
                    mask = torch.tensor(probs_hat[y_idx] < p_thres)
                    mask_idx = torch.nonzero(mask).squeeze(-1)
                    confident_idx = torch.nonzero(mask == False).squeeze(-1)
                    mask_num = len(mask_idx)
                    mask_num_concat = 0
                    iter_decode = False
                    if (t1) % (block_len * subsample * (1+cache_len) // 2) == 0 and t1 > 0:

                        iter_decode = True
                        if cache_len > 0:
                            hist_length = sum(
                                y_in_length_cache[-cache_len:])
                        else:
                            hist_length = 0
                        mask_hist = mask_cache[len(mask_cache) - hist_length:]

                        mask_idx_concat = torch.nonzero(
                            torch.cat([mask_hist, mask])).squeeze(-1)
                        mask_num_concat = len(mask_idx_concat)

                    y_in = torch.zeros(
                        1, len(y_idx), dtype=torch.long) + self.mask_token

                    y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

                    logging.info("init msk:{}".format(n2s(y_in[0].tolist())))
                    
                    # iterative decoding
                    if mask_num_concat > 0 and recog_args.maskctc_n_iterations > 0 and iter_decode:
                        K = recog_args.maskctc_n_iterations

                        y_in_hist = y_in_cache[:, y_in_cache.size(
                            1) - hist_length:].clone()
                        y_in_hist[0][mask_hist] = self.mask_token
                        y_in_concat = torch.cat([y_in_hist, y_in], dim=-1)
                        num_iter = K if mask_num_concat >= K and K > 0 else mask_num_concat

                        # h_pad_in plus previous h_pad
                        if t1 >= bh*subsample*(3+cache_len):
                            h_pad_in = self.encoder(x[t1-bh*subsample*(3+cache_len):t1-1, :]
                                                    .unsqueeze(0), None, None, None)[0][:, block_len:, :]
                        else:
                            h_pad_in = self.encoder(x[:t1-1, :]
                                                    .unsqueeze(0), None, None, None)[0]

                        for n_iter in range(num_iter-1):

                            pred, _ = self.decoder(
                                y_in_concat, None, h_pad_in, None)
                            pred_score, pred_id = pred[0][mask_idx_concat].max(
                                dim=-1)

                            cand = torch.topk(
                                pred_score, mask_num_concat // num_iter, -1)[1]

                            y_in_concat[0][mask_idx_concat[cand]
                                           ] = pred_id[cand]
                            mask_idx_concat = torch.nonzero(
                                y_in_concat[0] == self.mask_token).squeeze(-1)
                        # edit leftover masks (|masks| < mask_num // num_iter)
                        pred, pred_mask = self.decoder(
                            y_in_concat, None, h_pad_in, None)
                        pred_score, pred_id = pred[0].max(dim=-1)
                        y_in_concat[0][mask_idx_concat] = pred_id[mask_idx_concat]
                        logging.info("msk:{}".format(
                            n2s(y_in_concat[0].tolist())))

                        # history decoder output
                        ret[len(ret)-hist_length:] = y_in_concat.tolist()[0]
                        # pdb.set_trace()
                    else:
                        pred_score = torch.zeros(
                            (y_in.size(1),), dtype=torch.float) + 100
                        y_in[0] = y_hat[y_idx]
                        ret = ret + y_in.tolist()[0]
                        logging.info("ctc:{}".format(n2s(y_in[0].tolist())))
                    mask_cache = torch.cat(
                        [mask_cache, mask], dim=0)
                    pred_score_cache = torch.cat(
                        [pred_score_cache, pred_score])
                    y_in_cache = torch.cat([y_in_cache, y_in], dim=-1)
                    y_in_length_cache.append(y_in.size(1))
                    t1 = t+1

                if t == x.size(0) - 1:
                    break

        else:
            h = self.encoder(x.unsqueeze(0), None, None, None)[0]
            # greedy ctc outputs
            ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)

            y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
            y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

            # calculate token-level ctc probabilities by taking
            # the maximum probability of consecutive frames with
            # the same ctc symbols
            probs_hat = []
            cnt = 0
            for i, y in enumerate(y_hat.tolist()):
                probs_hat.append(-1)
                while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                    if probs_hat[i] < ctc_probs[0][cnt]:
                        probs_hat[i] = ctc_probs[0][cnt].item()
                    cnt += 1
            probs_hat = torch.from_numpy(numpy.array(probs_hat))

            # mask ctc outputs based on ctc probabilities
            p_thres = recog_args.maskctc_probability_threshold
            mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
            confident_idx = torch.nonzero(
                probs_hat[y_idx] >= p_thres).squeeze(-1)
            mask_num = len(mask_idx)

            y_in = torch.zeros(
                1, len(y_idx), dtype=torch.long) + self.mask_token
            y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

            logging.info("ctc:{}".format(n2s(y_in[0].tolist())))

            # iterative decoding
            if not mask_num == 0 and recog_args.maskctc_n_iterations > 0:
                K = recog_args.maskctc_n_iterations
                num_iter = K if mask_num >= K and K > 0 else mask_num
                for t in range(num_iter - 1):
                    pred, _ = self.decoder(y_in, None, h, None)

                    pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                    cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                    y_in[0][mask_idx[cand]] = pred_id[cand]
                    mask_idx = torch.nonzero(
                        y_in[0] == self.mask_token).squeeze(-1)

                    logging.info("msk:{}".format(n2s(y_in[0].tolist())))

                # predict leftover masks (|masks| < mask_num // num_iter)
                pred, pred_mask = self.decoder(y_in, None, h, None)
                y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

            else:
                y_in[0] = y_hat[y_idx]

            ret = y_in.tolist()[0]

        logging.warning("running time:{}".format(time.time()-start))
        hyp = {"score": 0.0, "yseq": [self.sos] + ret + [self.eos]}
        return [hyp]
