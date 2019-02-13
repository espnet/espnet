#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import math
import os

import chainer
import kaldiio
import numpy as np
import torch

from chainer import training
from chainer.training import extensions

from espnet.tts.tts_utils import get_dimensions
from espnet.tts.tts_utils import make_args_batchset
from espnet.tts.tts_utils import prepare_trainer

from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.e2e_tts import Tacotron2
from espnet.nets.pytorch_backend.e2e_tts import Tacotron2Loss

from espnet.transform.transformation import using_transform_config

from espnet.utils.io_utils import LoadInputsAndTargets

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import get_model_conf
from espnet.utils.training.train_utils import load_jsons
from espnet.utils.training.train_utils import write_conf

from espnet.utils.pytorch_utils import get_iterators
from espnet.utils.pytorch_utils import torch_load
from espnet.utils.pytorch_utils import warn_if_no_cuda


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
                 use_second_target=False,
                 preprocess_conf=None):
        self.return_targets = return_targets
        self.use_speaker_embedding = use_speaker_embedding
        self.use_second_target = use_second_target
        self.load_inputs_and_targets = LoadInputsAndTargets(
            mode='tts',
            use_speaker_embedding=use_speaker_embedding,
            use_second_target=use_second_target,
            preprocess_conf=preprocess_conf)

    def transform(self, item):
        # load batch
        xs, ys, spembs, spcs = self.load_inputs_and_targets(item)
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

    warn_if_no_cuda()

    idim, odim = get_dimensions(args)

    write_conf(args, idim, odim)

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
    converter = CustomConverter(return_targets=True,
                                use_speaker_embedding=args.use_speaker_embedding,
                                use_second_target=args.use_cbhg,
                                preprocess_conf=args.preprocess_conf)

    train_json, valid_json = load_jsons(args)

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if use_sortagrad:
        args.batch_sort_key = "input"
    # make minibatch list (variable length)
    train_batchset = make_args_batchset(train_json, args)
    valid_batchset = make_args_batchset(valid_json, args)

    train_iter, valid_iter = get_iterators(train_batchset, valid_batchset, converter, args.n_iter_processes,
                                           use_sortagrad=use_sortagrad)

    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, converter, device)
    evaluator = CustomEvaluator(model, valid_iter, reporter, converter, device)
    att_fig_converter = CustomConverter(False, args.use_speaker_embedding, args.preprocess_conf)
    trainer = prepare_trainer(updater, evaluator, att_fig_converter, model, [train_iter], valid_json, args, device)

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


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

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='tts', load_input=False, sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf)

    with torch.no_grad(), kaldiio.WriteHelper('ark,scp:{o}.ark,{o}.scp'.format(o=args.out)) as f:
        for idx, utt_id in enumerate(js.keys()):
            batch = [(utt_id, js[utt_id])]
            with using_transform_config({'train': False}):
                data = load_inputs_and_targets(batch)
            if train_args.use_speaker_embedding:
                spemb = data[1][0]
                spemb = torch.FloatTensor(spemb).to(device)
            else:
                spemb = None
            x = data[0][0]
            x = torch.LongTensor(x).to(device)

            # decode and write
            outs, _, _ = tacotron2.inference(x, args, spemb)
            if outs.size(0) == x.size(0) * args.maxlenratio:
                logging.warning("output length reaches maximum length (%s)." % utt_id)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            f[utt_id] = outs.cpu().numpy()
