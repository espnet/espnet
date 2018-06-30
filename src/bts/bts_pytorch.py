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
from torch.autograd import Variable

import kaldi_io_py

from asr_utils import PlotAttentionReport
from e2e_asr_backtrans import Tacotron2
from e2e_asr_backtrans import Tacotron2Loss
from e2e_asr_backtrans import torch_is_old

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
        # TODO(kan-bayashi): need to be fixed in pytorch v4
        if not torch_is_old:
            torch.set_grad_enabled(False)
        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                # convert to torch tensor
                batch = self.converter(batch, self.model.training)
                self.model(*batch)
            summary.add(observation)
        # TODO(kan-bayashi): need to be fixed in pytorch v4
        if not torch_is_old:
            torch.set_grad_enabled(True)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    '''CUSTOM UPDATER FOR TACOTRON2 TRAINING'''

    def __init__(self, model, grad_clip, train_iter, optimizer, converter):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.converter = converter
        # TODO(kan-bayashi): need to be fixed in pytorch v4
        if torch_is_old:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        batch = self.converter(train_iter.next(), self.model.training)

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

    def __init__(self, device, return_targets=True):
        self.device = device
        self.return_targets = return_targets

    def __call__(self, batch, is_training=True):
        # batch should be located in list
        assert len(batch) == 1
        batch = batch[0]

        # get eos
        eos = str(int(batch[0][1]['output'][0]['shape'][1]) - 1)

        # get target features and input character sequence
        xs = [b[1]['output'][0]['tokenid'].split() + [eos] for b in batch]
        ys = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]

        # remove empty sequence and get sort along with length
        filtered_idx = filter(lambda i: len(xs[i]) > 0, range(len(ys)))
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(xs[i]))
        xs = [np.fromiter(map(int, xs[i]), dtype=np.int64) for i in sorted_idx]
        ys = [ys[i] for i in sorted_idx]

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.fromiter((x.shape[0] for x in xs), dtype=np.int64))
        olens = torch.from_numpy(np.fromiter((y.shape[0] for y in ys), dtype=np.int64))

        # perform padding and convert to tensor
        xs = torch.from_numpy(pad_ndarray_list(xs, 0)).long()
        ys = torch.from_numpy(pad_ndarray_list(ys, 0)).float()

        # make labels for stop prediction
        labels = ys.new(ys.size(0), ys.size(1)).zero_()
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1

        # TODO(kan-bayashi): need to be fixed in pytorch v4
        if torch_is_old:
            xs = Variable(xs, volatile=not is_training)
            ys = Variable(ys, volatile=not is_training)
            labels = Variable(labels, volatile=not is_training)

        if sum(self.device) >= 0:
            xs = xs.cuda()
            ys = ys.cuda()
            labels = labels.cuda()

        if self.return_targets:
            return xs, ilens, ys, labels, olens
        else:
            return xs, ilens, ys


def pad_ndarray_list(batch, pad_value=float("nan")):
    """FUNCTION TO PERFORM PADDING OF NDARRAY LIST

    :param list batch: list of the ndarray [(T_1, D), (T_2, D), ..., (T_B, D)]
    :param float pad_value: value to pad
    :return: padded batch with the shape (B, Tmax, D)
    :rtype: ndarray
    """
    bs = len(batch)
    maxlen = max([b.shape[0] for b in batch])
    if len(batch[0].shape) >= 2:
        batch_pad = np.zeros((bs, maxlen) + batch[0].shape[1:])
    else:
        batch_pad = np.zeros((bs, maxlen))
    batch_pad.fill(pad_value)
    for i, b in enumerate(batch):
        batch_pad[i, :b.shape[0]] = b

    return batch_pad


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
            ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
            olen = int(sorted_data[start][1]['output'][0]['shape'][0])
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
            ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
            olen = int(sorted_data[start][1]['output'][0]['shape'][0])
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

    # define output activation function
    if args.output_activation is None:
        output_activation_fn = None
    elif hasattr(torch.nn.functional, args.output_activation):
        output_activation_fn = getattr(torch.nn.functional, args.output_activation)
    else:
        raise ValueError('there is no such an activation function. (%s)' % args.output_activation)

    # specify model architecture
    tacotron2 = Tacotron2(
        idim=idim,
        odim=odim,
        embed_dim=args.embed_dim,
        elayers=args.elayers,
        eunits=args.eunits,
        econv_layers=args.econv_layers,
        econv_chans=args.econv_chans,
        econv_filts=args.econv_filts,
        dlayers=args.dlayers,
        dunits=args.dunits,
        prenet_layers=args.prenet_layers,
        prenet_units=args.prenet_units,
        postnet_layers=args.postnet_layers,
        postnet_chans=args.postnet_chans,
        postnet_filts=args.postnet_filts,
        output_activation_fn=output_activation_fn,
        adim=args.adim,
        aconv_chans=args.aconv_chans,
        aconv_filts=args.aconv_filts,
        cumulate_att_w=args.cumulate_att_w,
        use_batch_norm=args.use_batch_norm,
        use_concate=args.use_concate,
        dropout=args.dropout_rate,
        zoneout=args.zoneout_rate)
    logging.info(tacotron2)

    # Set gpu
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        tacotron2.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        tacotron2 = torch.nn.DataParallel(tacotron2, device_ids=gpu_id)
        tacotron2.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * ngpu))
        args.batch_size *= ngpu
    else:
        gpu_id = [-1]

    # define loss
    model = Tacotron2Loss(
        model=tacotron2,
        use_masking=args.use_masking,
        bce_pos_weight=args.bce_pos_weight)
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
    converter = CustomConverter(gpu_id)
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
        data = sorted(valid_json.items()[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        trainer.extend(PlotAttentionReport(
            tacotron2, data, args.outdir + '/att_ws',
            CustomConverter(gpu_id, False), True), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/mse_loss', 'validation/main/mse_loss',
                                          'main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='loss.png'))
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
    report_keys = ['epoch', 'iteration', 'elapsed_time', 'main/loss', 'main/mse_loss', 'main/bce_loss',
                   'validation/main/loss', 'validation/main/mse_loss', 'validation/main/bce_loss']
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

    # define output activation function
    if hasattr(train_args, 'output_activation'):
        if train_args.output_activation is None:
            output_activation_fn = None
        elif hasattr(torch.nn.functional, train_args.output_activation):
            output_activation_fn = getattr(torch.nn.functional, train_args.output_activation)
        else:
            raise ValueError('there is no such an activation function. (%s)' % train_args.output_activation)
    else:
        output_activation_fn = None

    # define model
    tacotron2 = Tacotron2(
        idim=idim,
        odim=odim,
        embed_dim=train_args.embed_dim,
        elayers=train_args.elayers,
        eunits=train_args.eunits,
        econv_layers=train_args.econv_layers,
        econv_chans=train_args.econv_chans,
        econv_filts=train_args.econv_filts,
        dlayers=train_args.dlayers,
        dunits=train_args.dunits,
        prenet_layers=train_args.prenet_layers,
        prenet_units=train_args.prenet_units,
        postnet_layers=train_args.postnet_layers,
        postnet_chans=train_args.postnet_chans,
        postnet_filts=train_args.postnet_filts,
        adim=train_args.adim,
        aconv_chans=train_args.aconv_chans,
        aconv_filts=train_args.aconv_filts,
        output_activation_fn=output_activation_fn,
        cumulate_att_w=train_args.cumulate_att_w,
        use_batch_norm=train_args.use_batch_norm,
        use_concate=train_args.use_concate,
        dropout=train_args.dropout_rate,
        zoneout=train_args.zoneout_rate,
        threshold=args.threshold,
        maxlenratio=args.maxlenratio,
        minlenratio=args.minlenratio,
    )
    eos = str(tacotron2.idim - 1)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    tacotron2.load_state_dict(
        torch.load(args.model, map_location=lambda storage, loc: storage))
    tacotron2.eval()

    # Set gpu
    ngpu = args.ngpu
    if ngpu >= 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        tacotron2.cuda()
    else:
        gpu_id = [-1]

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # chech direcitory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
    arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.out, args.out)
    # TODO(kan-bayashi): need to be fixed in pytorch v4
    if not torch_is_old:
        torch.set_grad_enabled(False)
    with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
        for idx, utt_id in enumerate(js.keys()):
            x = js[utt_id]['output'][0]['tokenid'].split() + [eos]
            x = np.fromiter(map(int, x), dtype=np.int64)
            x = torch.from_numpy(x)
            # TODO(kan-bayashi): need to be fixed in pytorch v4
            if torch_is_old:
                x = Variable(x, volatile=True)
            if args.ngpu > 0:
                x = x.cuda()
            outs, probs, att_ws = tacotron2.inference(x)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            kaldi_io_py.write_mat(f, outs.cpu().numpy(), utt_id)
