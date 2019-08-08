#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import math

import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders_rnnt import decoder_for

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, loss, cer, wer):
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('loss:' + str(loss))
        reporter.report({'loss': loss}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (namespace): argument Namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.rnnt_mode = args.rnnt_mode
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()

        # note that eos is the same as sos (equivalent ID)
        self.sos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype.endswith("p") and not args.etype.startswith("vgg"):
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        if args.use_frontend:
            # Relative importing because of using python3 syntax
            from espnet.nets.pytorch_backend.frontends.feature_transform \
                import feature_transform_for
            from espnet.nets.pytorch_backend.frontends.frontend \
                import frontend_for
            
            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None
        
        # encoder
        self.enc = encoder_for(args, idim, self.subsample)

        if args.rnnt_mode == 1:
            # attention
            self.att = att_for(args)
            # decoder
            self.dec = decoder_for(args, odim, self.sos, self.att)
        else:
            # prediction
            self.dec = decoder_for(args, odim, self.sos)
        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'nbest': args.nbest,
                          'space': args.sym_space}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False

        self.logzero = -10000000000.0
        self.rnnlm = None
        self.loss = None

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """

        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        
        if self.rnnt_mode == 1:
            # embed weight ~ Normal(0, 1)
            self.dec.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in range(len(self.dec.decoder)):
                set_forget_bias_to_one(self.dec.decoder[l].bias_ih)
        else:
            self.dec.embed.weight.data.normal_(0, 1)

            
    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
               loss (torch.Tensor): transducer loss value
        """

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens
        
        # 1. encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # 2. decoder
        loss = self.dec(hs_pad, hlens, ys_pad)

        # 3. compute cer/wer
        ## Note: not recommended outside debugging right now,
        ## the training time is hugely impacted.
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
        else:
            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []

            batchsize =  int(hs_pad.size(0))
            batch_nbest = []

            for b in six.moves.range(batchsize):
                nbest_hyps = self.dec.recognize_beam(hs_pad[b], self.recog_args)
                batch_nbest.append(nbest_hyps)
                
            y_hats = [nbest_hyp[0]['yseq'][1:] for nbest_hyp in batch_nbest]
            
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars)) 
            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)
                
        self.loss = loss
        loss_data = float(self.loss)
        
        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E recognize

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
           y (list): n-best decoding results
        """
        
        prev = self.training
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. Encoder
        h, _, _ = self.enc(hs, hlens)

        # 2. Decoder (pred+joint or att-dec+joint)
        if recog_args.search_type == 'greedy':
            y = self.dec.recognize(h[0], recog_args)
        else:
            y = self.dec.recognize_beam(h[0], recog_args, rnnlm)

        if prev:
            self.train()

        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E recognize with batch

        Args:
            xs (ndarray): list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
           y (list): n-best decoding results
        """

        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # Subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(xs_pad, ilens)
            hs_pad, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))

        if recog_args.search_type == 'greedy':
            y = self.dec.recognize_batch(hs_pad, hlens, recog_args)
        else:
            y = self.dec.recognize_beam_batch(hs_pad, hlens, recog_args, rnnlm)
            
        if prev:
            self.train()

        return y
        
    def enhance(self, xs):
        """Forwarding only the frontend stage

        Args:
            xs (ndarray): input acoustic feature (T, C, F)
        
        Returns:
            enhanced (ndarray):
            mask (torch.Tensor):
            ilens (torch.Tensor): batch of lengths of input sequences (B)
        """

        if self.frontend is None:
            raise RuntimeError('Frontend does\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)

        if prev:
            self.train()

        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens
    
    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
            att_ws (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).
        """
        
        with torch.no_grad():
             # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens
            
            # encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws

    def subsample_frames(self, x):
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen
