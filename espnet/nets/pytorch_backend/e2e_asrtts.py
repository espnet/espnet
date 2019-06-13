#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math

import editdistance

import chainer
import numpy as np
import pickle
import six
import torch
import torch.nn.functional as F

# from chainer import reporter

from espnet.asr.asr_utils import torch_load
from espnet.nets.e2e_asr_common import label_smoothing_dist

from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders_asrtts import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

from espnet.nets.pytorch_backend.nets_utils import mask_by_length_and_multiply
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import set_requires_grad
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

from espnet.nets.pytorch_backend.e2e_tts import Tacotron2
from espnet.nets.pytorch_backend.e2e_tts import Tacotron2Loss

CTC_LOSS_THRESHOLD = 10000


def TacotronRewardLoss(tts_model_file, idim=None, odim=None, train_args=None,
                       use_masking=False, bce_pos_weight=20.0,
                       spk_embed_dim=None, update_asr_only=True, reporter=None):
    # TACOTRON CYCLE-CONSISTENT LOSS HERE
    # Define model
    tacotron2 = Tacotron2(
        idim=idim,
        odim=odim,
        args=train_args
    )
    if tts_model_file:
        # load trained model parameters
        logging.info('reading model parameters from ' + tts_model_file)

        torch_load(tts_model_file, tacotron2)
    else:
        logging.info("not using pretrained tacotron2 model")
    # Set to eval mode
    if update_asr_only:
        tacotron2.eval()
    # Define loss
    loss = Tacotron2Loss(
        model=tacotron2,
        use_masking=use_masking,
        bce_pos_weight=bce_pos_weight,
        reporter=reporter
    )
    if update_asr_only:
        loss.eval()
    loss.train_args = train_args
    return loss


def load_tacotron_loss(tts_model_conf, tts_model_file, args, reporter=None):
    # Read model
    if 'conf' in tts_model_conf:
        with open(tts_model_conf, 'rb') as f:
            idim_taco, odim_taco, train_args_taco = pickle.load(f)
    elif 'json' in tts_model_conf:
        from espnet.asr.asrtts_utils import get_model_conf
        idim_taco, odim_taco, train_args_taco = get_model_conf(tts_model_file, conf_path=tts_model_conf)
    if args.modify_output:
        import json
        with open(args.valid_json, 'rb') as f:
            valid_json = json.load(f)['utts']
        utts = list(valid_json.keys())
        idim_taco = int(valid_json[utts[0]]['output'][0]['shape'][1])
        from espnet.asr.asrtts_utils import remove_output_layer
        pretrained_model = remove_output_layer(torch.load(tts_model_file),
                                               idim_taco, args.eprojs, train_args_taco.embed_dim, 'tts')

        torch.save(pretrained_model, 'tmp.model')
        tts_model_file = 'tmp.model'
    # Load loss
    return TacotronRewardLoss(
        tts_model_file,
        idim=idim_taco,
        odim=odim_taco,
        train_args=train_args_taco,
        update_asr_only=args.update_asr_only,
        reporter=reporter
    )


class Tacotron2ASRLoss(torch.nn.Module):
    """TACOTRON2 ASR-LOSS FUNCTION

    :param torch.nn.Module model: tacotron2 model
    :param bool use_masking: whether to mask padded part in loss calculation
    :param float bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    :param bool report: Use reporter to log loss values (deafult true)
    :param bool reduce: Reduce the loss over the batch size
    """

    def __init__(self, tts_model, asr_model, args, reporter=None, weight=1.0):
        super(Tacotron2ASRLoss, self).__init__()
        self.tts_model = tts_model
        self.asr_model = asr_model
        self.reporter = reporter
        self.weight = weight
        self.generator = args.generator

    def forward(self, xs, ilens, ys, labels, olens=None, spembs=None, zero_att=False):
        """TACOTRON2 LOSS FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor labels: batch of the sequences of stop token labels (B, Lmax)
        :param list olens: batch of the lengths of each target (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: loss value
        :rtype: torch.Tensor
        """

        # generate feature sequences for a batch
        torch.set_grad_enabled(self.training)
        set_requires_grad(self.tts_model, True)
        if not zero_att:
            outs, logits, flens = self.tts_model.generate(xs, ilens, spembs)

        if not zero_att:
            if self.generator == 'tts':
                flens = sorted(flens, reverse=True)
                enc_outs, enc_flens, _ = self.asr_model.enc(outs, flens)
            # asr loss
            xslst = [xs[i, :ilens[i] - 1] for i in six.moves.range(enc_outs.size(0))]
            asr_loss, acc = self.asr_model.dec(enc_outs, enc_flens, xslst)
        else:
            xslst = [xs[i, :ilens[i] - 1] for i in six.moves.range(xs.size(0))]
            asr_loss, acc = self.asr_model.dec(xs, ilens, xslst, zero_att=zero_att)
        # calculate loss
        if not zero_att:
            flens = torch.LongTensor(flens)
            labels = torch.zeros(outs.size(0), outs.size(1), dtype=torch.float32)
            labels[torch.arange(outs.size(0), dtype=torch.int64), flens - 1] = 1
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels.cuda())
            loss = (asr_loss + bce_loss) * self.weight
            # report loss values for logging
            loss_data = loss.detach().cpu().numpy()
            asr_loss_data = asr_loss.detach().cpu().numpy()
            bce_loss_data = bce_loss.detach().cpu().numpy()
            logging.info("loss = %.3e (asr: %.3e, bce: %.3e)" % (loss_data,
                                                                 asr_loss_data,
                                                                 bce_loss_data))
        else:
            loss = asr_loss
            # report loss values for logging
            loss_data = loss.detach().cpu().numpy()
            asr_loss_data = asr_loss.detach().cpu().numpy()
            logging.info("asr_loss: %.3e" % (asr_loss_data))
        if self.reporter is not None and not zero_att:
            self.reporter.report(None, None, asr_loss_data, acc, None, None, None, None, bce_loss_data)
        elif self.reporter is not None and zero_att:
            self.reporter.report(None, None, asr_loss_data, acc, None, None, None, None, None)
        return loss


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(self, idim, odim, args, predictor=None, loss_fn=None, rnnlm=None,
                 softargmax=False, asr2tts=False, reporter=None):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.use_speaker_embedding = args.use_speaker_embedding
        self.n_samples_per_input = args.n_samples_per_input
        self.policy_gradient = args.policy_gradient
        self.sample_scaling = args.sample_scaling
        self.softargmax = softargmax
        self.generator = args.generator
        self.asr2tts = asr2tts
        self.rnnlm = rnnlm
        self.rnnloss = args.rnnloss
        self.reporter = reporter
        self.update_asr_only = args.update_asr_only
        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

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

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        self.enc = encoder_for(args, idim, self.subsample)
        # attention
        self.att = att_for(args)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

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
                elif data.dim() in (3, 4):
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
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad, spembs=None):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        if self.asr2tts:
            # sample output sequence with the current model
            gen_out = self.predictor.generate(xs_pad, ys_pad,
                                              n_samples_per_input=self.n_samples_per_input)
            loss_ctc, loss_att, ys, ygen, y_all, ylens, _, _ = gen_out
            if self.generator == 'tts':
                hpad = xs_pad
                ilens = np.fromiter((xx.shape[0] for xx in xs_pad), dtype=np.int64)
                hpad, hlens = mask_by_length_and_multiply(hpad, torch.tensor(ilens),
                                                          0, self.n_samples_per_input)
                if self.use_speaker_embedding:
                    onelens = np.fromiter((1 for xx in spembs), dtype=np.int64)
                    spembs, _ = mask_by_length_and_multiply(spembs.unsqueeze(1),
                                                            torch.tensor(onelens), 0, self.n_samples_per_input)
                    spembs = spembs.squeeze(1)
            if self.policy_gradient:  # use log posterior probs
                weight = self.sample_scaling * -loss_att
            else:  # use posterior probs
                weight = (self.sample_scaling * -loss_att).view(len(xs_pad),
                                                                self.n_samples_per_input)
                weight = torch.nn.Softmax(dim=1)(weight)
                weight = weight.view(-1)
            ylens, indices = torch.sort(ylens, descending=True)
            ygen = ygen[indices]
            hpad = hpad[indices]
            hlens = hlens[indices]
            weight = weight[indices]
            if self.rnnlm:
                with torch.no_grad():
                    sos = ygen.data.new([self.predictor.sos] * len(ygen))
                    rnnlm_state, lmz = self.rnnlm.predictor(None, sos)
                    lm_loss = F.cross_entropy(lmz, ygen[:, 0], reduction='none')
                    for i in six.moves.range(1, ylens[0]):
                        rnnlm_state, lmz = self.rnnlm.predictor(rnnlm_state, ygen[:, i - 1])
                        if self.rnnloss == 'ce':
                            loss_i = F.cross_entropy(lmz, ygen[:, i], reduction='none')
                        lm_loss += loss_i
                    lm_loss = lm_loss / ylens.sum().type_as(lm_loss)
            else:
                lm_loss = None
            # Weighted loss
            if self.loss_fn is not None:
                # make labels for stop prediction
                labels = hpad.new(hpad.size(0), hpad.size(1)).zero_()
                for i, l in enumerate(hlens):
                    labels[i, l - 1:] = 1
                labels = to_device(self, labels)
                # compute taco loss
                if self.softargmax:
                    y_all = torch.stack(y_all).reshape(hpad.size(0), len(y_all), -1)
                    # x_taco = (y_all, ylens, hpad, labels, None)
                    x_taco = (y_all, ylens, hpad, labels, hlens)
                else:
                    # x_taco = (ygen, ylens, hpad, labels, None)
                    x_taco = (ygen, ylens, hpad, labels, hlens)
                # self.loss_fn.eval()
                with torch.set_grad_enabled(not self.update_asr_only):
                    if self.use_speaker_embedding:
                        taco_loss = self.loss_fn(*x_taco, spembs=spembs,
                                                 softargmax=self.softargmax, asrtts=True)  # .mean(2).mean(1)
                    else:
                        taco_loss = self.loss_fn(*x_taco, softargmax=self.softargmax,
                                                 asrtts=True)  # .mean(2).mean(1)
                if lm_loss is not None:
                    taco_loss += lm_loss * self.lm_loss_weight
                if self.update_asr_only:
                    loss = ((taco_loss - taco_loss.mean()) * weight).mean()
                else:
                    loss = ((taco_loss - taco_loss.mean()) * weight).mean() + taco_loss.mean()
                chainer.Reporter().report({'loss_mse': float(taco_loss.mean().cpu())})
            else:
                if lm_loss is not None:
                    loss = (-weight + lm_loss * self.lm_loss_weight).mean()
                else:
                    loss = -weight.mean()
            # Inform use
            if self.verbose > 0 and self.char_list is not None:
                for i in six.moves.range(len(xs_pad)):
                    for j in six.moves.range(self.n_samples_per_input):
                        k = i * self.n_samples_per_input + j
                        y_str = "".join([self.char_list[int(ygen[k, l])] for l in range(ylens[k])])
                        if self.loss_fn is not None:
                            logging.info("generation[%d]: %.4f %.4f " % (k, weight[k],
                                                                         taco_loss[k]) + y_str)
                        else:
                            logging.info("generation[%d]: %.4f " % (k,
                                                                    weight[k].item()) + y_str)
            loss_att = loss
            acc = 0.0
            self.acc = acc

        else:
            # 1. Encoder
            hs_pad, hlens, _ = self.enc(xs_pad, ilens)

            # 2. attention loss
            loss_att, acc = self.dec(hs_pad, hlens, ys_pad)
            self.acc = acc

        # 4. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(hs_pad, torch.tensor(hlens), lpz,
                                                       self.recog_args, self.char_list,
                                                       self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
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

        self.loss = loss_att
        loss_att_data = float(loss_att)
        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            if self.loss_fn is not None:
                self.reporter.report(None, loss_att_data, None, None, None, None,
                                     None, float(taco_loss.mean().cpu()), None)
            else:
                self.reporter.report(loss_att_data, None, None, None, acc, cer, wer, None, None)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        # Note(kamo): In order to work with DataParallel, on pytorch==0.4,
        # the return value must be torch.CudaTensor, or tuple/list/dict of it.
        # Neither CPUTensor nor float/int value can be used
        # because NCCL communicates between GPU devices.
        device = next(self.parameters()).device
        acc = torch.tensor([acc], device=device) if acc is not None else None
        cer = torch.tensor([cer], device=device)
        wer = torch.tensor([wer], device=device)
        return self.loss, loss_att, acc, cer, wer

    def generate(self, xs, ys, n_samples_per_input=10, topk=0, maxlenratio=1.0,
                 minlenratio=0.3, freeze_encoder=False, return_encoder_states=False,
                 oracle=False):
        '''E2E generate

        :param data:
        :return:
        '''
        torch.set_grad_enabled(self.training)

        # utt list of frame x dim
        # xs = [d[1]['feat'] for d in data]

        # remove 0-output-length utterances
        # tids = [d[1]['output'][0]['tokenid'].split() for d in data]
        # filtered_index = filter(lambda i: len(tids[i]) > 0, range(len(xs)))
        filtered_index = filter(lambda i: len(xs[i]) > 0, range(len(xs)))
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
        if len(sorted_index) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(sorted_index)))
        xs = [xs[i] for i in sorted_index]

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        if self.training:
            hs = [to_device(self, xx) for xx in xs]
        else:
            hs = [to_device(self, xx) for xx in xs]

        # 1. encoder
        xpad = pad_list(hs, 0)
        hpad, hlens, _ = self.enc(xpad, ilens)
        # expand encoder states by n_sample_per_input
        hpad, hlens = mask_by_length_and_multiply(hpad, hlens, 0, n_samples_per_input)
        if freeze_encoder:
            logging.warn("Encoder is frozen")
            new_hpad = hpad.detach()
            del hpad
            hpad = new_hpad

        # set_requires_grad(self.dec, False)
        if oracle:
            ys = [to_device(self, torch.from_numpy(y)) for y in ys for id in range(n_samples_per_input)]
            loss_att, y_gen, ylens = self.dec.gen_oracle(hpad, hlens, ys)
        else:
            loss_att, ys, y_gen, ylens, y_all = self.dec.generate(hpad, hlens,
                                                                  topk=topk,
                                                                  maxlenratio=maxlenratio,
                                                                  minlenratio=minlenratio)
            # 3. CTC loss
            ys = [to_device(self, torch.from_numpy(y)) for y in ys]

        loss_ctc = None

        y_gen = to_device(self, torch.from_numpy(y_gen))
        ylens = to_device(self, torch.from_numpy(ylens))
        return loss_ctc, loss_att, ys, y_gen, y_all, ylens, xpad, ilens

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        prev = self.training
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 1. encoder
        hs, _, _ = self.enc(hs, ilens)

        # calculate log P(z_t|X) for CTC scores
        lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()

        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 1. encoder
        hs_pad, hlens, _ = self.enc(xs_pad, ilens)

        # calculate log P(z_t|X) for CTC scores
        lpz = None

        # 2. decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """

        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. Decoder
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
