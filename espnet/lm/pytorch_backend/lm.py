#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

import copy
import json
import logging
import numpy as np

import torch
import torch.nn as nn

from chainer import Chain
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

from espnet.lm.lm_utils import compute_perplexity
from espnet.lm.lm_utils import count_tokens
from espnet.lm.lm_utils import load_dataset
from espnet.lm.lm_utils import MakeSymlinkToBestModel
from espnet.lm.lm_utils import ParallelSentenceIterator
from espnet.lm.lm_utils import read_tokens
from espnet.nets.lm_interface import LMInterface

from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop


# dummy module to use chainer's trainer
class Reporter(Chain):
    def report(self, loss):
        pass


def concat_examples(batch, device=None, padding=None):
    """Custom concat_examples for pytorch

    :param np.ndarray batch: The batch to concatenate
    :param int device: The device to send to
    :param Tuple[int,int] padding: The padding to use
    :return: (inputs, targets)
    :rtype (torch.Tensor, torch.Tensor)
    """
    x, t = convert.concat_examples(batch, padding=padding)
    x = torch.from_numpy(x)
    t = torch.from_numpy(t)
    if device is not None and device >= 0:
        x = x.cuda(device)
        t = t.cuda(device)
    return x, t


class BPTTUpdater(training.StandardUpdater):
    """An updater for a pytorch LM

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param LMInterface model : The model to update
    :param optimizer:
    :param int device : The device id
    :param int gradclip : The gradient clipping value to use
    """

    def __init__(self, train_iter, model, optimizer, device, gradclip=None):
        super(BPTTUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.device = device
        self.gradclip = gradclip

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        # Progress the dataset iterator for sentences at each iteration.
        batch = train_iter.__next__()
        # Concatenate the token IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        x, t = concat_examples(batch, device=self.device, padding=(0, -100))
        loss, logp, count = self.model(x, t)
        reporter.report({'loss': float(logp.sum())}, optimizer.target)
        reporter.report({'count': int(count.sum())}, optimizer.target)
        # update
        self.model.zero_grad()  # Clear the parameter gradients
        loss.mean().backward()  # Backprop
        if self.gradclip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip)
        optimizer.step()  # Update the parameters


class LMEvaluator(BaseEvaluator):
    """A custom evaluator for a pytorch LM

    :param chainer.dataset.Iterator val_iter : The validation iterator
    :param LMInterface eval_model : The model to evaluate
    :param chainer.Reporter reporter : The observations reporter
    :param int device : The device id to use
    """

    def __init__(self, val_iter, eval_model, reporter, device):
        super(LMEvaluator, self).__init__(
            val_iter, reporter, device=-1)
        self.model = eval_model
        self.device = device

    def evaluate(self):
        val_iter = self.get_iterator('main')
        logp = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in copy.copy(val_iter):
                x, t = concat_examples(batch, device=self.device, padding=(0, -100))
                _, l, c = self.model(x, t)
                logp += float(l.sum())
                count += int(c.sum())
        self.model.train()
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'loss': logp / count}, self.model.reporter)
        return observation


def train(args, model_class):
    """Train with the given args

    :param Namespace args: The program arguments
    :param type model_class: LMInterface class for training
    """
    assert issubclass(model_class, LMInterface), "model should implement LMInterface"
    # display torch version
    logging.info('torch version = ' + torch.__version__)

    set_deterministic_pytorch(args)

    # check cuda and cudnn availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get special label ids
    unk = args.char_list_dict['<unk>']
    eos = args.char_list_dict['<eos>']
    # read tokens as a sequence of sentences
    val, n_val_tokens, n_val_oovs = load_dataset(args.valid_label, args.char_list_dict, args.dump_hdf5_path)
    train, n_train_tokens, n_train_oovs = load_dataset(args.train_label, args.char_list_dict, args.dump_hdf5_path)
    logging.info('#vocab = ' + str(args.n_vocab))
    logging.info('#sentences in the training data = ' + str(len(train)))
    logging.info('#tokens in the training data = ' + str(n_train_tokens))
    logging.info('oov rate in the training data = %.2f %%' % (n_train_oovs / n_train_tokens * 100))
    logging.info('#sentences in the validation data = ' + str(len(val)))
    logging.info('#tokens in the validation data = ' + str(n_val_tokens))
    logging.info('oov rate in the validation data = %.2f %%' % (n_val_oovs / n_val_tokens * 100))

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # Create the dataset iterators
    batch_size = args.batchsize * max(args.ngpu, 1)
    if batch_size > args.batchsize:
        logging.info(f'batch size is automatically increased ({args.batchsize} -> {batch_size})')
    train_iter = ParallelSentenceIterator(train, batch_size,
                                          max_length=args.maxlen, sos=eos, eos=eos, shuffle=not use_sortagrad)
    val_iter = ParallelSentenceIterator(val, batch_size,
                                        max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
    logging.info('#iterations per epoch = ' + str(len(train_iter.batch_indices)))
    logging.info('#total iterations = ' + str(args.epoch * len(train_iter.batch_indices)))
    # Prepare an RNNLM model
    model = model_class(args.n_vocab, args)
    reporter = Reporter()
    if args.ngpu > 0:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu))).cuda()
        gpu_id = 0
    else:
        gpu_id = -1
    setattr(model, "reporter", reporter)

    # Save model conf to json
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    # Set up an optimizer
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    updater = BPTTUpdater(train_iter, model, optimizer, gpu_id, gradclip=args.gradclip)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)
    trainer.extend(LMEvaluator(val_iter, model, reporter, device=gpu_id))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(args.report_interval_iters, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity', 'elapsed_time']
    ), trigger=(args.report_interval_iters, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    # Save best models
    trainer.extend(torch_snapshot(filename='snapshot.ep.{.updater.epoch}'))
    trainer.extend(snapshot_object(model, 'rnnlm.model.{.updater.epoch}'))
    # T.Hori: MinValueTrigger should be used, but it fails when resuming
    trainer.extend(MakeSymlinkToBestModel('validation/main/loss', 'rnnlm.model'))

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epoch, 'epoch'))
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    set_early_stop(trainer, args, is_lm=True)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer), trigger=(args.report_interval_iters, 'iteration'))

    trainer.run()
    check_early_stop(trainer, args.epoch)

    # compute perplexity for test set
    if args.test_label:
        logging.info('test the best model')
        torch_load(args.outdir + '/rnnlm.model.best', model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info('#sentences in the test data = ' + str(len(test)))
        logging.info('#tokens in the test data = ' + str(n_test_tokens))
        logging.info('oov rate in the test data = %.2f %%' % (n_test_oovs / n_test_tokens * 100))
        test_iter = ParallelSentenceIterator(test, batch_size,
                                             max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
        evaluator = LMEvaluator(test_iter, model, reporter, device=gpu_id)
        result = evaluator()
        logging.info('test perplexity: ' + str(np.exp(float(result['main/loss']))))
