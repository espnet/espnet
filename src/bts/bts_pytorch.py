#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import math
import os
import pickle

import chainer
import numpy as np
import torch

from chainer import training
from chainer.training import extensions

import kaldi_io_py
import lazy_io

from e2e_asr_attctc_th import pad_list
from e2e_asr_backtrans import Tacotron2
from e2e_asr_backtrans import Tacotron2Loss

import matplotlib
matplotlib.use('Agg')

MAX_SAVE_ATT_W = 5


def make_batchset(data, batch_size, max_length_in, max_length_out, num_batches=0):
    # sort it by output lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['ilen']), reverse=True)
    logging.info('# utts: ' + str(len(sorted_data)))
    # change batchsize depending on the input and output length
    minibatch = []
    start = 0
    while True:
        ilen = int(sorted_data[start][1]['olen'])  # reverse
        olen = int(sorted_data[start][1]['ilen'])  # reverse
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
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch


def prepare_batch(batch, reader, eos):
    # get target features and input character sequence
    xs = [b[1]['tokenid'].split() + [str(eos)] for b in batch]
    ys = [reader[b[0].encode('ascii', 'ignore')] for b in batch]

    # remove empty sequence and get sort along with length
    filtered_idx = filter(lambda i: len(xs[i]) > 0, range(len(ys)))
    sorted_idx = sorted(filtered_idx, key=lambda i: -len(xs[i]))
    xs = [np.fromiter(map(int, xs[i]), dtype=np.int64) for i in sorted_idx]
    ys = [ys[i] for i in sorted_idx]

    # get list of lengths
    ilens = torch.from_numpy(np.fromiter((x.shape[0] for x in xs), dtype=np.int64))
    olens = torch.from_numpy(np.fromiter((y.shape[0] for y in ys), dtype=np.int64))

    # perform padding and convert to tensor
    xs = torch.from_numpy(pad_list(xs, 0)).long()
    ys = torch.from_numpy(pad_list(ys, 0)).float()

    # make labels for stop prediction
    labels = ys.new_zeros((ys.size(0), ys.size(1)))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1  # l or l-1?

    if torch.cuda.is_available():
        xs = xs.cuda()
        ys = ys.cuda()
        labels = labels.cuda()

    return xs, ilens, ys, labels, olens


class PytorchSeqEvaluaterKaldi(extensions.Evaluator):
    '''Custom evaluater with Kaldi reader for pytorch'''

    def __init__(self, model, criterion, iterator, target, reader, device, outdir):
        super(PytorchSeqEvaluaterKaldi, self).__init__(
            iterator, target, device=device)
        self.reader = reader
        self.criterion = criterion
        self.model = model
        self.outdir = outdir
        self.epoch = 0
        self.num_gpu = len(device)
        self.eos = model.idim - 1 if self.num_gpu == 1 else model.module.idim - 1

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
            for idx, batch in enumerate(it):
                observation = {}
                with chainer.reporter.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    # batch only has one minibatch utterance, which is specified by batch[0]
                    xs, ilens, ys, labels, olens = prepare_batch(batch[0], self.reader, self.eos)
                    after_outs, before_outs, logits, att_ws = self.model(xs, ilens, ys)
                    self.criterion(after_outs, before_outs, logits, ys, labels, olens)

                # save visialized attention weight
                if idx < MAX_SAVE_ATT_W:
                    matplotlib.pyplot.imshow(att_ws[0].cpu().numpy(), aspect="auto")
                    matplotlib.pyplot.savefig(
                        self.outdir + "/att_w_idx%d_epoch%d.png" % (idx, self.epoch))
                    matplotlib.pyplot.close()

                summary.add(observation)

        self.epoch += 1
        self.model.train()

        return summary.compute_mean()


class PytorchSeqUpdaterKaldi(training.StandardUpdater):
    '''Custom updater with Kaldi reader for pytorch'''

    def __init__(self, model, criterion, grad_clip_threshold, train_iter, optimizer, reader, device):
        super(PytorchSeqUpdaterKaldi, self).__init__(
            train_iter, optimizer, device=None)
        self.model = model
        self.criterion = criterion
        self.reader = reader
        self.grad_clip_threshold = grad_clip_threshold
        self.num_gpu = len(device)
        self.eos = model.idim - 1 if self.num_gpu == 1 else model.module.idim - 1

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.__next__()

        # read scp files
        # x: original json with loaded features
        #    will be converted to chainer variable later
        # batch only has one minibatch utterance, which is specified by batch[0]
        if len(batch[0]) < self.num_gpu:
            logging.warning('batch size is less than number of gpus. Ignored')
            return
        xs, ilens, ys, labels, olens = prepare_batch(batch[0], self.reader, self.eos)

        # Compute the loss at this time step and accumulate it
        after_outs, before_outs, logits, att_ws = self.model(xs, ilens, ys)
        loss = self.criterion(after_outs, before_outs, logits, ys, labels, olens)
        optimizer.zero_grad()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
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
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    # reverse input and output dimension
    idim = int(valid_json[utts[0]]['odim'])
    odim = int(valid_json[utts[0]]['idim'])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify model architecture
    model = Tacotron2(
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
        adim=args.adim,
        aconv_chans=args.aconv_chans,
        aconv_filts=args.aconv_filts,
        cumulate_att_w=args.cumulate_att_w,
        use_batch_norm=args.use_batch_norm,
        use_concate=args.use_concate,
        dropout=args.dropout_rate)
    criterion = Tacotron2Loss(
        use_masking=args.use_masking,
        bce_pos_weight=args.bce_pos_weight)
    logging.info(model)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        pickle.dump((idim, odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    reporter = criterion.reporter
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model = torch.nn.DataParallel(model, device_ids=gpu_id)
        model.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    else:
        gpu_id = [-1]

    # Setup an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, eps=args.eps,
        weight_decay=args.weight_decay)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_label, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(valid, 1, repeat=False, shuffle=False)

    # prepare Kaldi reader
    train_reader = lazy_io.read_dict_scp(args.train_feat)
    valid_reader = lazy_io.read_dict_scp(args.valid_feat)

    # Set up a trainer
    updater = PytorchSeqUpdaterKaldi(
        model, criterion, args.grad_clip, train_iter, optimizer, train_reader, gpu_id)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        if ngpu > 1:
            model.module.load_state_dict(torch.load(args.outdir + '/model.loss.best'))
        else:
            model.load_state_dict(torch.load(args.outdir + '/model.loss.best'))
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(PytorchSeqEvaluaterKaldi(
        model, criterion, valid_iter, reporter, valid_reader, gpu_id, args.outdir))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/mse_loss', 'validation/main/mse_loss',
                                          'main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/mse_loss', 'validation/main/mse_loss'],
                                         'epoch', file_name='mse_loss.png'))
    trainer.extend(extensions.PlotReport(['main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='bce_loss.png'))

    # Save best models
    def torch_save(path, _):
        if ngpu > 1:
            torch.save(model.module.state_dict(), path)
            torch.save(model.module, path + ".pkl")
        else:
            torch.save(model.state_dict(), path)
            torch.save(model, path + ".pkl")

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
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
    '''Generate encoder states'''
    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from ' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    # show argments
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # define model
    model = Tacotron2(idim=idim,
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
                      cumulate_att_w=train_args.cumulate_att_w,
                      use_batch_norm=train_args.use_batch_norm,
                      use_concate=train_args.use_concate,
                      dropout=train_args.dropout_rate)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    model.load_state_dict(
        torch.load(args.model, map_location=lambda storage, loc: storage))
    model.eval()

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # Set gpu
    ngpu = args.ngpu
    if ngpu >= 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    else:
        gpu_id = [-1]

    # read json data
    with open(args.label, 'rb') as f:
        js = json.load(f)['utts']

    # chech direcitory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
    torch.set_grad_enabled(False)
    arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.out, args.out)
    with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
        for idx, utt_id in enumerate(js.keys()):
            x = js[utt_id]['tokenid'].split() + [str(model.idim - 1)]
            x = np.fromiter(map(int, x), dtype=np.int64)
            x = torch.from_numpy(x)
            if args.ngpu > 0:
                x = x.cuda()
            outs, probs, att_ws = model.inference(x)
            logging.info("(%d/%d) %s (size:%d->%d)" % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            kaldi_io_py.write_mat(f, outs.cpu().numpy(), utt_id)
