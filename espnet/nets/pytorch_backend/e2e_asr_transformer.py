# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from distutils.util import strtobool
import logging
import math

import torch

from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy

from .transformer.attention import MIN_VALUE
from .transformer.attention import MultiHeadedAttention
from .transformer.decoder import Decoder
from .transformer.encoder import Encoder
from .transformer.layer_norm import LayerNorm
from .transformer.loss import LabelSmoothing


def add_arguments(parser):
    group = parser.add_argument_group("transformer model setting")
    group.add_argument("--transformer-init", type=str, default="pytorch",
                       choices=["pytorch", "xavier_uniform", "xavier_normal",
                                "kaiming_uniform", "kaiming_normal"],
                       help='how to initialize transformer parameters')
    group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                       choices=["conv2d", "linear", "embed"],
                       help='transformer input layer type')
    group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                       help='dropout in transformer attention. use --dropout-rate if None is set')
    group.add_argument('--transformer-lr', default=10.0, type=float,
                       help='Initial value of learning rate')
    group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                       help='optimizer warmup steps')
    group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                       help='normalize loss by length')
    return parser


def subsequent_mask(size, device="cpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


class E2E(torch.nn.Module):
    def __init__(self, idim, odim, args, ignore_id=-1):
        super(E2E, self).__init__()
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(idim, args)
        self.decoder = Decoder(odim, args)
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.reporter = Reporter()

        # self.lsm_weight = a
        self.criterion = LabelSmoothing(self.odim, self.ignore_id, args.lsm_weight,
                                        args.transformer_length_normalized_loss)
        # self.char_list = args.char_list
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.recog_args = None  # unused
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = ctc_for(odim, args)
        else:
            self.ctc = None

    def reset_parameters(self, args):
        if args.transformer_init == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if args.transformer_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.transformer_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.transformer_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.transformer_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.transformer_init)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()

    def add_sos_eos(self, ys_pad):
        from espnet.nets.pytorch_backend.nets_utils import pad_list
        eos = ys_pad.new([self.eos])
        sos = ys_pad.new([self.sos])
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        return pad_list(ys_in, self.eos), pad_list(ys_out, self.ignore_id)

    def target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != self.ignore_id
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, xs_pad, ilens, ys_pad, dec_in=None):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # forward decoder
        if dec_in is None:
            ys_in_pad, ys_out_pad = self.add_sos_eos(ys_pad)
        else:
            ys_in_pad = dec_in
            ys_out_pad = ys_pad
        ys_mask = self.target_mask(ys_in_pad)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        self.pred_pad = pred_pad

        # compute loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                          ignore_label=self.ignore_id)

        # TODO(karita) show predected text
        # TODO(karita) calculate these stats
        cer, cer_ctc, wer = 0.0, 0.0, 0.0
        if self.ctc is None:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)

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
            self.reporter.report(loss_ctc_data, loss_att_data, acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        # Note(kamo): In order to work with DataParallel, on pytorch==0.4,
        # the return value must be torch.CudaTensor, or tuple/list/dict of it.
        # Neither CPUTensor nor float/int value can be used
        # because NCCL communicates between GPU devices.
        device = next(self.parameters()).device
        acc = torch.tensor([acc], device=device)
        cer = torch.tensor([cer], device=device)
        wer = torch.tensor([wer], device=device)
        return self.loss, loss_ctc, loss_att, acc, cer, wer

    def recognize(self, feat, recog_args, char_list=None, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        feat = torch.as_tensor(feat).unsqueeze(0)
        feat_len = [feat.size(1)]
        mask = (~make_pad_mask(feat_len)).to(feat.device).unsqueeze(-2)
        enc_output, mask = self.encoder(feat, mask)

        # TODO(karita) support CTC, LM, lpz
        if recog_args.beam_size == 1:
            logging.info("use greedy search implementation")
            ys = torch.full((1, 1), self.sos).long()
            score = torch.zeros(1)
            maxlen = feat.size(1) + 1
            for step in range(maxlen):
                ys_mask = subsequent_mask(step + 1).unsqueeze(0)
                out, _ = self.decoder(ys, ys_mask, enc_output, mask)
                prob = torch.log_softmax(out[:, -1], dim=-1)  # (batch, token)
                max_prob, next_id = prob.max(dim=1)  # (batch, token)
                score += max_prob
                if step == maxlen - 1:
                    next_id[0] = self.eos

                ys = torch.cat((ys, next_id.unsqueeze(1)), dim=1)
                if next_id[0].item() == self.eos:
                    break
            y = [{"score": score, "yseq": ys[0].tolist()}]
        else:
            # TODO(karita) maxlen minlen
            logging.info("use beam search implementation")

            # TODO(karita) batch decoding
            n_beam = recog_args.beam_size
            enc_output = enc_output
            score = torch.zeros(1)
            if recog_args.maxlenratio == 0:
                maxlen = feat.size(1) + 1
            else:
                maxlen = max(1, int(recog_args.maxlenratio * feat.size(1)))
            minlen = int(recog_args.minlenratio * feat.size(1))
            logging.info('max output length: ' + str(maxlen))
            logging.info('min output length: ' + str(minlen))

            # TODO(karita) GPU decoding (I think it is almost ready)
            ended = torch.full((n_beam,), False, dtype=torch.uint8)
            ys = torch.full((1, 1), self.sos, dtype=torch.int64)
            n_local_beam = n_beam
            for step in range(maxlen):
                # forward
                ys_mask = subsequent_mask(step + 1).unsqueeze(0)
                if step == 1:
                    enc_output = enc_output.expand(n_beam, *enc_output.shape)
                out, _ = self.decoder(ys[:, :step + 1], ys_mask, enc_output, mask)
                prob = torch.log_softmax(out[:, -1], dim=-1)  # (beam, token)
                if step > 0:
                    prob = prob.masked_fill(ended.unsqueeze(-1), MIN_VALUE)

                # prune
                if n_local_beam == -1:
                    n_local_beam = prob.size(1)
                local_score, local_id = prob.topk(n_local_beam, dim=1)  # (1 or beam, local_beam)
                if step > 0:
                    local_score *= (~ended).float().unsqueeze(1)
                global_score = score.unsqueeze(-1) + local_score  # (1 or beam, local_beam)
                global_score, global_id = global_score.view(-1).topk(n_beam)  # (beam)
                local_hyp = global_id % n_local_beam  # NOTE assume global_score is contiguous
                if step > 0:
                    prev_hyp = global_id // n_local_beam  # NOTE ditto
                else:
                    prev_hyp = torch.zeros(n_beam, dtype=torch.int64)
                top_tokens = local_id[prev_hyp, local_hyp]  # (beam)
                logging.info("global_id: " + str(global_id))
                logging.info("prev_hyp:  " + str(prev_hyp))
                logging.info("local_hyp: " + str(local_hyp))
                logging.info("top-tokens: " + str(top_tokens))

                # update stats
                if step > 0:
                    score_diff = prob.masked_fill(ended.unsqueeze(-1), 0)[prev_hyp, top_tokens]
                else:
                    score_diff = prob[0, top_tokens]
                score = score + score_diff
                score += (~ended).float() * recog_args.penalty
                new_ys = torch.empty((n_beam, ys.size(1) + 1), dtype=torch.int64)
                new_ys[:, :-1] = ys[prev_hyp]
                new_ys[:, -1] = top_tokens.masked_fill(ended, self.eos) if step > 0 else top_tokens
                ys = new_ys
                ended = ended[prev_hyp]
                if char_list is None:
                    logging.info("beam: " + str(ys))
                else:
                    for i in range(n_beam):
                        s = "".join((char_list[int(c)].replace("<space>", " ") for c in ys[i]))
                        logging.info("beam {}: {}".format(i, s))
                if step > minlen:
                    ended |= top_tokens == self.eos
                if ended.all():
                    break

            ys[:, -1] = self.eos
            yseq = [[self.sos] for i in range(n_beam)]
            for i in range(n_beam):
                for y in ys[i, 1:]:
                    yseq[i].append(int(y))
                    if y == self.eos:
                        break
            y = [{"score": score[i].item(), "yseq": yseq[i]} for i in range(n_beam)]
            y = sorted(y, key=lambda x: x["score"], reverse=True)[:min(len(y), recog_args.nbest)]
        self.training = prev
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn
        return ret
