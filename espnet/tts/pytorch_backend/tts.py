#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import math
import os

import chainer
import numpy as np
import torch

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

import kaldi_io_py

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.e2e_tts import Tacotron2
from espnet.nets.pytorch_backend.e2e_tts import Tacotron2Loss
from espnet.tts.tts_utils import load_inputs_and_targets
from espnet.tts.tts_utils import make_batchset

from espnet.bin.bin_utils import set_deterministic_pytorch

import matplotlib

from espnet.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

matplotlib.use('Agg')


REPORT_INTERVAL = 100


class CustomEvaluator(extensions.Evaluator):
    """Custom Evaluator for Tacotron2 training

    :param torch.nn.Model model : The model to evaluate
    :param chainer.dataset.Iterator iterator : The validation iterator
    :param target :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device to use
    """

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

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
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom updater for Tacotron2 training

    :param torch.nn.Module model: The model to update
    :param float grad_clip : The gradient clipping value to use
    :param chainer.dataset.Iterator train_iter : The training iterator
    :param optimizer :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device to use
    """

    def __init__(self, model, grad_clip, train_iter, optimizer, converter, device):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.converter = converter
        self.device = device
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        # compute loss and gradient
        loss = self.model(*x)
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
    """Custom converter for Tacotron2 training

    :param bool return_targets:
    :param bool use_speaker_embedding:
    :param bool use_second_target:
    """

    def __init__(self,
                 return_targets=True,
                 use_speaker_embedding=False,
                 use_second_target=False):
        self.return_targets = return_targets
        self.use_speaker_embedding = use_speaker_embedding
        self.use_second_target = use_second_target

    def transform(self, item):
        # load batch
        xs, ys, spembs, spcs = load_inputs_and_targets(
            item, self.use_speaker_embedding, self.use_second_target)

        # added eos into input sequence
        eos = int(item[0][1]['output'][0]['shape'][1]) - 1
        xs = [np.append(x, eos) for x in xs]

        return xs, ys, spembs, spcs

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        inputs_and_targets = batch[0]

        # parse inputs and targets
        xs, ys, spembs, spcs = inputs_and_targets

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1.0

        # load second target
        if spcs is not None:
            spcs = pad_list([torch.from_numpy(spc).float() for spc in spcs], 0).to(device)

        # load speaker embedding
        if spembs is not None:
            spembs = torch.from_numpy(np.array(spembs)).float().to(device)

        if self.return_targets:
            return xs, ilens, ys, labels, olens, spembs, spcs
        else:
            return xs, ilens, ys, spembs


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)

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
    if args.use_cbhg:
        args.spc_dim = int(valid_json[utts[0]]['input'][1]['shape'][1])
    if args.use_speaker_embedding:
        args.spk_embed_dim = int(valid_json[utts[0]]['input'][1]['shape'][0])
    else:
        args.spk_embed_dim = None
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
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

    # Setup a converter
    converter = CustomConverter(True, args.use_speaker_embedding, args.use_cbhg)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train_batchset = make_batchset(train_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out,
                                   args.minibatches, args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    valid_batchset = make_batchset(valid_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out,
                                   args.minibatches, args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(train_batchset, converter.transform),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid_batchset, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = chainer.iterators.SerialIterator(
            TransformDataset(train_batchset, converter.transform),
            batch_size=1)
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid_batchset, converter.transform),
            batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, converter, device)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save snapshot for each epoch
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # Save best models
    trainer.extend(extensions.snapshot_object(tacotron2, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

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
            converter=CustomConverter(False, args.use_speaker_embedding),
            device=device, reverse=True), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    plot_keys = ['main/loss', 'validation/main/loss',
                 'main/l1_loss', 'validation/main/l1_loss',
                 'main/mse_loss', 'validation/main/mse_loss',
                 'main/bce_loss', 'validation/main/bce_loss']
    trainer.extend(extensions.PlotReport(['main/l1_loss', 'validation/main/l1_loss'],
                                         'epoch', file_name='l1_loss.png'))
    trainer.extend(extensions.PlotReport(['main/mse_loss', 'validation/main/mse_loss'],
                                         'epoch', file_name='mse_loss.png'))
    trainer.extend(extensions.PlotReport(['main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='bce_loss.png'))
    if args.use_cbhg:
        plot_keys += ['main/cbhg_l1_loss', 'validation/main/cbhg_l1_loss',
                      'main/cbhg_mse_loss', 'validation/main/cbhg_mse_loss']
        trainer.extend(extensions.PlotReport(['main/cbhg_l1_loss', 'validation/main/cbhg_l1_loss'],
                                             'epoch', file_name='cbhg_l1_loss.png'))
        trainer.extend(extensions.PlotReport(['main/cbhg_mse_loss', 'validation/main/cbhg_mse_loss'],
                                             'epoch', file_name='cbhg_mse_loss.png'))
    trainer.extend(extensions.PlotReport(plot_keys, 'epoch', file_name='loss.png'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = plot_keys[:]
    report_keys[0:0] = ['epoch', 'iteration', 'elapsed_time']
    trainer.extend(extensions.PrintReport(report_keys), trigger=(REPORT_INTERVAL, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer))

    # Run the training
    trainer.run()


def decode(args):
    """Decode with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    tacotron2 = Tacotron2(idim, odim, train_args)
    eos = str(tacotron2.idim - 1)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, tacotron2)
    tacotron2.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    tacotron2 = tacotron2.to(device)

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # check directory
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
                logging.warning("output length reaches maximum length (%s)." % utt_id)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            kaldi_io_py.write_mat(f, outs.cpu().numpy(), utt_id)
