#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import numpy as np
import six

import chainer
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L

# for classifier link
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from chainer import training
from chainer.training import extensions

from espnet.lm.lm_utils import compute_perplexity
from espnet.lm.lm_utils import count_tokens
from espnet.lm.lm_utils import MakeSymlinkToBestModel
from espnet.lm.lm_utils import ParallelSentenceIterator
from espnet.lm.lm_utils import read_tokens
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.lm_interface import LMInterface

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop


class BPTTUpdater(training.updaters.StandardUpdater):
    """An updater for a chainer LM

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param optimizer:
    :param int device : The device id
    """

    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)

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
        x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
        loss, nll, count = optimizer.target(x, t)
        loss_data = float(loss.data)
        nll_data = float(nll.data)
        reporter.report({'loss': loss_data}, optimizer.target)
        reporter.report({'nll': nll_data}, optimizer.target)
        reporter.report({'count': count}, optimizer.target)
        logging.info('loss {:.04f}, nll {:.04f}, counts {}'.format(loss_data, nll_data, count))
        # update
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


class LMEvaluator(BaseEvaluator):
    """A custom evaluator for a chainer LM

    :param chainer.dataset.Iterator val_iter : The validation iterator
    :param eval_model : The model to evaluate
    :param int device : The device id to use
    """

    def __init__(self, val_iter, eval_model, device):
        super(LMEvaluator, self).__init__(
            val_iter, eval_model, device=device)

    def evaluate(self):
        val_iter = self.get_iterator('main')
        target = self.get_target('main')
        loss = 0
        nll = 0
        count = 0
        for batch in copy.copy(val_iter):
            x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
            l, n, c = target(x, t)
            loss += float(l.data)
            nll += float(n.data)
            count += int(c)
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'loss': loss}, target)
            reporter.report({'nll': nll}, target)
            reporter.report({'count': count}, target)
        return observation


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    model_class = dynamic_import_lm(args.model_module, args.backend)
    assert issubclass(model_class, LMInterface), "model should implement LMInterface"
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    set_deterministic_chainer(args)

    # check cuda and cudnn availability
    if not chainer.cuda.available:
        logging.warning('cuda is not available')
    if not chainer.cuda.cudnn_enabled:
        logging.warning('cudnn is not available')

    # get special label ids
    unk = args.char_list_dict['<unk>']
    eos = args.char_list_dict['<eos>']
    # read tokens as a sequence of sentences
    train = read_tokens(args.train_label, args.char_list_dict)
    val = read_tokens(args.valid_label, args.char_list_dict)
    # count tokens
    n_train_tokens, n_train_oovs = count_tokens(train, unk)
    n_val_tokens, n_val_oovs = count_tokens(val, unk)
    logging.info('#vocab = ' + str(args.n_vocab))
    logging.info('#sentences in the training data = ' + str(len(train)))
    logging.info('#tokens in the training data = ' + str(n_train_tokens))
    logging.info('oov rate in the training data = %.2f %%' % (n_train_oovs / n_train_tokens * 100))
    logging.info('#sentences in the validation data = ' + str(len(val)))
    logging.info('#tokens in the validation data = ' + str(n_val_tokens))
    logging.info('oov rate in the validation data = %.2f %%' % (n_val_oovs / n_val_tokens * 100))

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0

    # Create the dataset iterators
    train_iter = ParallelSentenceIterator(train, args.batchsize,
                                          max_length=args.maxlen, sos=eos, eos=eos, shuffle=not use_sortagrad)
    val_iter = ParallelSentenceIterator(val, args.batchsize,
                                        max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
    logging.info('#iterations per epoch = ' + str(len(train_iter.batch_indices)))
    logging.info('#total iterations = ' + str(args.epoch * len(train_iter.batch_indices)))
    # Prepare an RNNLM model
    model = model_class(args.n_vocab, args)
    if args.ngpu > 1:
        logging.warning("currently, multi-gpu is not supported. use single gpu.")
    if args.ngpu > 0:
        # Make the specified GPU current
        gpu_id = 0
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    else:
        gpu_id = -1

    # Save model conf to json
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    # Set up an optimizer
    if args.opt == 'sgd':
        optimizer = chainer.optimizers.SGD(lr=1.0)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_iter, optimizer, gpu_id)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)
    trainer.extend(LMEvaluator(val_iter, model, device=gpu_id))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(args.report_interval_iters, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity', 'elapsed_time']
    ), trigger=(args.report_interval_iters, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    trainer.extend(extensions.snapshot(filename='snapshot.ep.{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model, 'rnnlm.model.{.updater.epoch}'))
    # MEMO(Hori): wants to use MinValueTrigger, but it seems to fail in resuming
    trainer.extend(MakeSymlinkToBestModel('validation/main/loss', 'rnnlm.model'))

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epoch, 'epoch'))

    if args.resume:
        logging.info('resumed from %s' % args.resume)
        chainer.serializers.load_npz(args.resume, trainer)

    set_early_stop(trainer, args, is_lm=True)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer), trigger=(args.report_interval_iters, 'iteration'))

    trainer.run()
    check_early_stop(trainer, args.epoch)

    # compute perplexity for test set
    if args.test_label:
        logging.info('test the best model')
        chainer.serializers.load_npz(args.outdir + '/rnnlm.model.best', model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info('#sentences in the test data = ' + str(len(test)))
        logging.info('#tokens in the test data = ' + str(n_test_tokens))
        logging.info('oov rate in the test data = %.2f %%' % (n_test_oovs / n_test_tokens * 100))
        test_iter = ParallelSentenceIterator(test, args.batchsize,
                                             max_length=args.maxlen, sos=eos, eos=eos, repeat=False)
        evaluator = LMEvaluator(test_iter, model, device=gpu_id)
        with chainer.using_config('train', False):
            result = evaluator()
        logging.info('test perplexity: ' + str(np.exp(float(result['main/loss']))))
