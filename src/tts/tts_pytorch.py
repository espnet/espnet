#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import math
import os
import pickle
import random

import chainer
import numpy as np
import torch

from chainer import training
from chainer.training import extensions

import kaldi_io_py

from asr_utils import pad_ndarray_list
from asr_utils import PlotAttentionReport
from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss

import matplotlib
matplotlib.use('Agg')


class CustomEvaluator(extensions.Evaluator):
    '''CUSTOM EVALUATER FOR TACOTRON2 TRAINING'''

    def __init__(self, model, iterator, target, converter):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with chainer.reporter.report_scope(observation):
                    # convert to torch tensor
                    batch = self.converter(batch)
                    self.model(*batch)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    '''CUSTOM UPDATER FOR TACOTRON2 TRAINING'''

    def __init__(self, model, grad_clip, train_iter, optimizer, converter):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.converter = converter
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        batch = self.converter(train_iter.next())

        # compute loss and gradient
        loss = self.model(*batch)
        optimizer.zero_grad()
        loss.backward()

        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip)
        logging.debug('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()


class CustomConverter(object):
    '''CUSTOM CONVERTER FOR TACOTRON2'''

    def __init__(self, device, return_targets=True, use_speaker_embedding=False):
        self.device = device
        self.return_targets = return_targets
        self.use_speaker_embedding = use_speaker_embedding

    def __call__(self, batch):
        # batch should be located in list
        assert len(batch) == 1
        batch = batch[0]

        # get eos
        eos = str(int(batch[0][1]['output'][0]['shape'][1]) - 1)

        # get target features and input character sequence
        xs = [b[1]['output'][0]['tokenid'].split() + [eos] for b in batch]
        ys = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]

        # remove empty sequence and get sort along with length
        filtered_idx = filter(lambda i: len(xs[i]) > 0, range(len(xs)))
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(xs[i]))
        xs = [np.fromiter(map(int, xs[i]), dtype=np.int64) for i in sorted_idx]
        ys = [ys[i] for i in sorted_idx]

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.LongTensor([x.shape[0] for x in xs]).to(self.device)
        olens = torch.LongTensor([y.shape[0] for y in ys]).to(self.device)

        # perform padding and convert to tensor
        xs = torch.LongTensor(pad_ndarray_list(xs, 0)).to(self.device)
        ys = torch.FloatTensor(pad_ndarray_list(ys, 0)).to(self.device)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1

        # load speaker embedding
        if self.use_speaker_embedding:
            spembs = [kaldi_io_py.read_vec_flt(b[1]['input'][1]['feat']) for b in batch]
            spembs = [spembs[i] for i in sorted_idx]
            spembs = torch.FloatTensor(np.array(spembs)).to(self.device)
        else:
            spembs = None

        if self.return_targets:
            return xs, ilens, ys, labels, olens, spembs
        else:
            return xs, ilens, ys, spembs


def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, batch_sort_key=None):
    minibatch = []
    start = 0
    if batch_sort_key is None:
        logging.info('use shuffled batch.')
        shuffled_data = random.sample(data.items(), len(data.items()))
        logging.info('# utts: ' + str(len(shuffled_data)))
        while True:
            end = min(len(shuffled_data), start + batch_size)
            minibatch.append(shuffled_data[start:end])
            if end == len(shuffled_data):
                break
            start = end
    elif batch_sort_key == 'input':
        logging.info('use batch sorted by input length and adaptive batch size.')
        # sort it by output lengths (long to short)
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['output'][0]['shape'][0]), reverse=True)
        logging.info('# utts: ' + str(len(sorted_data)))
        # change batchsize depending on the input and output length
        while True:
            # input and output are reversed due to the use of same json as asr
            ilen = int(sorted_data[start][1]['output'][0]['shape'][0])
            olen = int(sorted_data[start][1]['input'][0]['shape'][0])
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
            # if ilen = 1000 and max_length_in = 800
            # then b = batchsize / 2
            # and max(1, .) avoids batchsize = 0
            b = max(1, int(batch_size / (1 + factor)))
            end = min(len(sorted_data), start + b)
            minibatch.append(sorted_data[start:end])
            if end == len(sorted_data):
                break
            start = end
    elif batch_sort_key == 'output':
        logging.info('use batch sorted by output length and adaptive batch size.')
        # sort it by output lengths (long to short)
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=True)
        logging.info('# utts: ' + str(len(sorted_data)))
        # change batchsize depending on the input and output length
        while True:
            # input and output are reversed due to the use of same json as asr
            ilen = int(sorted_data[start][1]['output'][0]['shape'][0])
            olen = int(sorted_data[start][1]['input'][0]['shape'][0])
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
            # if ilen = 1000 and max_length_in = 800
            # then b = batchsize / 2
            # and max(1, .) avoids batchsize = 0
            b = max(1, int(batch_size / (1 + factor)))
            end = min(len(sorted_data), start + b)
            minibatch.append(sorted_data[start:end])
            if end == len(sorted_data):
                break
            start = end
    else:
        ValueError('batch_sort_key should be selected from None, input, and output.')

    # for debugging
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch


def torch_save(path, model):
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    if hasattr(model, 'module'):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))


def train(args):
    '''RUN TRAINING'''
    # seed setting
    torch.manual_seed(args.seed)

    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())

    # reverse input and output dimension
    idim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    if args.use_speaker_embedding:
        args.spk_embed_dim = int(valid_json[utts[0]]['input'][1]['shape'][0])
    else:
        args.spk_embed_dim = None
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        pickle.dump((idim, odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    tacotron2 = Tacotron2(idim, odim, args)
    logging.info(tacotron2)

    # check the use of multi-gpu
    if args.ngpu > 1:
        tacotron2 = torch.nn.DataParallel(tacotron2, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    tacotron2 = tacotron2.to(device)

    # define loss
    model = Tacotron2Loss(tacotron2, args.use_masking, args.bce_pos_weight)
    reporter = model.reporter

    # Setup an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, eps=args.eps,
        weight_decay=args.weight_decay)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, 'target', reporter)
    setattr(optimizer, 'serialize', lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train_batchset = make_batchset(train_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out,
                                   args.minibatches, args.batch_sort_key)
    valid_batchset = make_batchset(valid_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out,
                                   args.minibatches, args.batch_sort_key)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train_batchset, 1)
    valid_iter = chainer.iterators.SerialIterator(valid_batchset, 1, repeat=False, shuffle=False)

    # Set up a trainer
    converter = CustomConverter(device, True, args.use_speaker_embedding)
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, converter)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('restored from %s' % args.resume)
        chainer.serializers.load_npz(args.resume, trainer)
        torch_load(args.outdir + '/model.ep.%d' % trainer.updater.epoch, tacotron2)
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(filename='snapshot.ep.{.updater.epoch}'), trigger=(1, 'epoch'))

    # Save attention figure for each epoch
    if args.num_save_attention > 0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(tacotron2, "module"):
            att_vis_fn = tacotron2.module.calculate_all_attentions
        else:
            att_vis_fn = tacotron2.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, data, args.outdir + '/att_ws',
            CustomConverter(device, False, args.use_speaker_embedding), True), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/l1_loss', 'validation/main/l1_loss',
                                          'main/mse_loss', 'validation/main/mse_loss',
                                          'main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/l1_loss', 'validation/main/l1_loss'],
                                         'epoch', file_name='l1_loss.png'))
    trainer.extend(extensions.PlotReport(['main/mse_loss', 'validation/main/mse_loss'],
                                         'epoch', file_name='mse_loss.png'))
    trainer.extend(extensions.PlotReport(['main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='bce_loss.png'))

    # Save model for each epoch
    trainer.extend(extensions.snapshot_object(
        tacotron2, 'model.ep.{.updater.epoch}', savefun=torch_save), trigger=(1, 'epoch'))

    # Save best models
    trainer.extend(extensions.snapshot_object(tacotron2, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    report_keys = ['epoch', 'iteration', 'elapsed_time',
                   'main/loss', 'main/l1_loss',
                   'main/mse_loss', 'main/bce_loss',
                   'validation/main/loss', 'validation/main/l1_loss',
                   'validation/main/mse_loss', 'validation/main/bce_loss']
    trainer.extend(extensions.PrintReport(report_keys), trigger=(100, 'iteration'))
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


def decode(args):
    '''RUN DECODING'''
    # read training config
    with open(args.model_conf, 'rb') as f:
        logging.info('reading a model config file from ' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    # show argments
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # define model
    tacotron2 = Tacotron2(idim, odim, train_args)
    eos = str(tacotron2.idim - 1)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    tacotron2.load_state_dict(
        torch.load(args.model, map_location=lambda storage, loc: storage))
    tacotron2.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    tacotron2 = tacotron2.to(device)

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # chech direcitory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
    arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.out, args.out)
    with torch.no_grad(), kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
        for idx, utt_id in enumerate(js.keys()):
            x = js[utt_id]['output'][0]['tokenid'].split() + [eos]
            x = np.fromiter(map(int, x), dtype=np.int64)
            x = torch.LongTensor(x).to(device)

            # get speaker embedding
            if train_args.use_speaker_embedding:
                spemb = kaldi_io_py.read_vec_flt(js[utt_id]['input'][1]['feat'])
                spemb = torch.FloatTensor(spemb).to(device)
            else:
                spemb = None

            # decode and write
            outs, _, _ = tacotron2.inference(x, args, spemb)
            if outs.size(0) == x.size(0) * args.maxlenratio:
                logging.warn("output length reaches maximum length (%s)." % utt_id)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            kaldi_io_py.write_mat(f, outs.cpu().numpy(), utt_id)
