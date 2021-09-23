# Copyright 2021 Johns Hopkins University (Nanxin Chen)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Align-Denoise non-autoregressive speech recognition model (pytorch).

See https://www.isca-speech.org/archive/pdfs/interspeech_2021/chen21q_interspeech.pdf

"""

import logging
import math

from distutils.util import strtobool
import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy


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
        E2E.add_aligndenoise_arguments(parser)

        return parser

    @staticmethod
    def add_aligndenoise_arguments(parser):
        """Add arguments for Align-Denoise model."""
        group = parser.add_argument_group("aligndenoise specific setting")

        group.add_argument(
            "--aligndenoise-beta",
            default=0.3,
            type=float,
        )
        group = add_arguments_conformer_common(group)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id)
        assert 0.0 <= self.mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"

        self.reset_parameters(args)

        self.betas = np.linspace(0.08, 0.2, 50)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.tensor(np.concatenate(([1], np.cumprod(self.alphas))), dtype=torch.float32)

        self.att_ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True, include_fc=False)

    def diffusion4(self, x, var, mask, noise_level=None):
        batch_size = x.shape[0]
        #Uniform index
        idx = torch.randint(0, len(self.betas), size=(batch_size,))
        lb = self.alphas_bar[idx + 1]
        ub = self.alphas_bar[idx]
        if not noise_level:
            noise_level = torch.rand(size=(batch_size,)) * (ub - lb) + lb
            noise_level = noise_level.unsqueeze(-1).unsqueeze(-1).to(device=x.device)
        noisy_x = torch.sqrt(noise_level) * x + torch.sqrt(1.0 - noise_level) * torch.randn_like(x) * var
        return x * mask + noisy_x * (1 - mask)
    
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
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, _ = self.encoder(xs_pad, src_mask)
        
        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        
        batch_size = xs_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
        ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
        cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

        
        if not self.training:
            self.ctc.softmax(hs_pad)

        # 2. forward decoder
        alignment = self.ctc.ctc_alignment_targets(hs_pad, hs_len, ys_pad).transpose(0, 1)
        label = alignment.argmax(dim=-1) * hs_mask[:, 0, :]

        var = torch.max(self.ctc.log_softmax(hs_pad).exp(), alignment * 0.3)
        mask = (self.ctc.argmax(hs_pad) == label).unsqueeze(-1).float()
        ys_in_pad = self.diffusion4(alignment, var, mask).argmax(dim=-1)
        ys_in_pad = ys_in_pad * hs_mask[:, 0, :].int()
        
        ys_mask = hs_mask
        
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        loss_att = self.att_ctc(pred_pad, hs_len, ys_pad)
        self.acc = 1.0 - self.error_calculator(pred_pad.argmax(dim=-1).data.cpu(), ys_pad.cpu(), is_ctc=True)
        
        self.pred_pad = pred_pad

        # 5. compute cer/wer
        cer, wer = None, None

        loss_att_data = float(loss_att)
        if loss_att_data > CTC_LOSS_THRESHOLD or math.isnan(loss_att_data):
            #import pickle
            #with open('tmp','wb') as f:
            #    pickle.dump((ys_pad.cpu(), alignment.cpu(), ys_in_pad.cpu(), pred_pad.cpu()), f)
            #raise Exception('debug')
            loss_att = 0

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            loss = loss_att
            loss_ctc_data = None
        elif alpha == 1:
            loss = loss_ctc
            loss_ctc_data = float(loss_ctc)
        else:
            loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_ctc_data = float(loss_ctc)

        loss_data = float(loss)
        if abs(loss_data) < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
            print("loss_att: {0} loss_ctc: {1}".format(loss_att_data, loss_ctc_data))
        self.loss = loss.detach().cpu()
        return loss


    def recognize(self, feat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """recognize feat.

        :param ndnarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list

        TODO(karita): do not recompute previous attention for faster decoding
        """
        xs = feat[np.newaxis, :]
        self.eval()
        xs_pad = torch.from_numpy(xs[0]).unsqueeze(0)
        ilens = torch.from_numpy(np.array([xs_pad.size(1)]))
        xs_mask = (make_non_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, _ = self.encoder(xs_pad, xs_mask)
        hs_pad_bak = hs_pad
        from itertools import groupby
        pred_pad = self.ctc.log_softmax(hs_pad).exp()

        char_list[0] = '_'
        for idx in range(len(char_list)):
            if char_list[idx] == '<space>':
                char_list[idx] = ' '
        logging.info('ctc : ' + ''.join([char_list[int(x)] for x in pred_pad.argmax(dim=-1).numpy().flatten().tolist()]))
        
        ys_in_pad = pred_pad.clone()
        ys_mask = None
        hs_len = hs_mask.view(ys_in_pad.size(0), -1).sum(1)

        betas = np.array([1.0])
        T = len(betas)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigma = np.sqrt(betas)

        for t in range(T - 1, -1, -1):
            logging.info('hypo: ' + ''.join([char_list[int(x)] for x in ys_in_pad.argmax(dim=-1).numpy().flatten().tolist()]))
            
            pred_pad, pred_mask = self.decoder(ys_in_pad.argmax(dim=-1), ys_mask, hs_pad, hs_mask)
            if self.decoder.output_layer is None:
                new_ys = self.ctc.log_softmax(pred_pad).exp()
            else:
                new_ys = pred_pad.softmax(dim=-1)
            logging.info('resu: ' + ''.join([char_list[int(x)] for x in new_ys.argmax(dim=-1).numpy().flatten().tolist()]))
            ys_in_pad = new_ys

        pred_pad, _ = self.decoder(ys_in_pad.argmax(dim=-1), ys_mask, hs_pad, hs_mask)
        if self.decoder.output_layer is None:
            pred_pad = self.ctc.log_softmax(pred_pad).exp()[0]
        else:
            pred_pad = pred_pad[0]#.exp()[0]
        idx = []
        ass = pred_pad.argmax(dim=-1).numpy().tolist()
        last = 0
        for i in range(pred_pad.size(0)):
            if ass[i] != 0 and ass[i]!= last:
                idx.append(i)
            elif ass[i] != 0 and ass[i] == last:
                if pred_pad[idx[-1]].max() < pred_pad[i].max():
                    idx[-1] = i
            last = ass[i]
            if ass[i] == 0:
                continue
        idx = np.array(idx)
        if len(idx) > 0:
            pred_pad = pred_pad[idx]
            return [{'score':0.0, 'yseq': [self.sos] + pred_pad.argmax(dim=-1).numpy().tolist() + [self.eos]}]
        else:
            return [{'score':0.0, 'yseq': [self.sos] + [self.eos]}]
