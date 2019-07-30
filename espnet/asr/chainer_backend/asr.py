#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import collections
import json
import logging
import math
import os
import six

# chainer related
import chainer

from chainer import cuda
from chainer import training
from chainer import Variable

from chainer.datasets import TransformDataset

from chainer.training import extensions
from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads

# espnet related
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

# rnnlm
import espnet.lm.chainer_backend.extlm as extlm_chainer
import espnet.lm.chainer_backend.lm as lm_chainer

# numpy related
import matplotlib
import numpy as np

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

matplotlib.use('Agg')

REPORT_INTERVAL = 100


# copied from https://github.com/chainer/chainer/blob/master/chainer/optimizer.py
def sum_sqnorm(arr):
    """Calculate the norm of the array.

    Args:
        arr (numpy.ndarray)

    Returns:
        Float: Sum of the norm calculated from the given array.

    """
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            if x is not None:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class CustomUpdater(training.StandardUpdater):
    """Custom updater for chainer.

    Args:
        train_iter (iterator | dict[str, iterator]): Dataset iterator for the
            training dataset. It can also be a dictionary that maps strings to
            iterators. If this is just an iterator, then the iterator is
            registered by the name ``'main'``.
        optimizer (optimizer | dict[str, optimizer]): Optimizer to update
            parameters. It can also be a dictionary that maps strings to
            optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter (espnet.asr.chainer_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.
        device (int or dict): The destination device info to send variables. In the
            case of cpu or single gpu, `device=-1 or 0`, respectively.
            In the case of multi-gpu, `device={"main":0, "sub_1": 1, ...}`.
        accum_grad (int):The number of gradient accumulation. if set to 2, the network
            parameters will be updated once in twice, i.e. actual batchsize will be doubled.

    """

    def __init__(self, train_iter, optimizer, converter, device, accum_grad=1):
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=device)
        self.forward_count = 0
        self.accum_grad = accum_grad
        self.start = True

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine for Custom Updater."""
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get batch and convert into variables
        batch = train_iter.next()
        x = self.converter(batch, self.device)
        if self.start:
            optimizer.target.cleargrads()
            self.start = False

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(*x) / self.accum_grad
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in optimizer.target.params(False)]))
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.update()
        optimizer.target.cleargrads()  # Clear the parameter gradients

    def update(self):
        self.update_core()
        if self.forward_count == 0:
            self.iteration += 1


class CustomParallelUpdater(training.updaters.MultiprocessParallelUpdater):
    """Custom Parallel Updater for chainer.

    Defines the main update routine.

    Args:
        train_iter (iterator | dict[str, iterator]): Dataset iterator for the
            training dataset. It can also be a dictionary that maps strings to
            iterators. If this is just an iterator, then the iterator is
            registered by the name ``'main'``.
        optimizer (optimizer | dict[str, optimizer]): Optimizer to update
            parameters. It can also be a dictionary that maps strings to
            optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter (espnet.asr.chainer_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.
        device (torch.device): Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        accum_grad (int):The number of gradient accumulation. if set to 2, the network
            parameters will be updated once in twice, i.e. actual batchsize will be doubled.

    """

    def __init__(self, train_iters, optimizer, converter, devices, accum_grad=1):
        super(CustomParallelUpdater, self).__init__(
            train_iters, optimizer, converter=converter, devices=devices)
        self.forward_count = 0
        self.accum_grad = accum_grad

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main Update routine of the custom parallel updater."""
        self.count += 1
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            from cupy.cuda import nccl
            # For reducing memory

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            x = self.converter(batch, self._devices[0])

            loss = self._master(*x) / self.accum_grad
            loss.backward()
            loss.unchain_backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl.NCCL_FLOAT,
                                 nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg

            # update parameters
            self.forward_count += 1
            if self.forward_count != self.accum_grad:
                return
            self.forward_count = 0
            # check gradient value
            grad_norm = np.sqrt(sum_sqnorm(
                [p.grad for p in optimizer.target.params(False)]))
            logging.info('grad norm={}'.format(grad_norm))

            # update
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.update()
            self._master.cleargrads()

            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT,
                                0, null_stream.ptr)

    def update(self):
        self.update_core()
        if self.forward_count == 0:
            self.iteration += 1


class CustomConverter(object):
    """Custom Converter.

    Args:
        subsampling_factor (int): The subsampling factor.

    """

    def __init__(self, subsampling_factor=1):
        self.subsampling_factor = subsampling_factor

    def __call__(self, batch, device):
        """Perform sabsampling.

        Args:
            batch (list): Batch that will be sabsampled.
            device (device): GPU device.

        Returns:
            chainer.Variable: xp.array that sabsampled from batch.
            xp.array: xp.array of the length of the mini-batches.
            chainer.Variable: xp.array that sabsampled from batch.

        """
        # set device
        xp = cuda.cupy if device != -1 else np

        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch made of lengths of input sequences
        ilens = [x.shape[0] for x in xs]

        # convert to Variable
        xs = [Variable(xp.array(x, dtype=xp.float32)) for x in xs]
        ilens = xp.array(ilens, dtype=xp.int32)
        ys = [Variable(xp.array(y, dtype=xp.int32)) for y in ys]

        return xs, ilens, ys


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    set_deterministic_chainer(args)

    # check cuda and cudnn availability
    if not chainer.cuda.available:
        logging.warning('cuda is not available')
    if not chainer.cuda.cudnn_enabled:
        logging.warning('cudnn is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # check attention type
    if args.atype not in ['noatt', 'dot', 'location']:
        raise NotImplementedError('chainer supports only noatt, dot, and location attention.')

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture
    logging.info('import model module: ' + args.model_module)
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, odim, args, flag_return=False)
    assert isinstance(model, ASRInterface)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = 0
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
        logging.info('single gpu calculation.')
    elif ngpu > 1:
        gpu_id = 0
        devices = {'main': gpu_id}
        for gid in six.moves.xrange(1, ngpu):
            devices['sub_%d' % gid] = gid
        logging.info('multi gpu calculation (#gpus = %d).' % ngpu)
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
    else:
        gpu_id = -1
        logging.info('cpu calculation')

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta(eps=args.eps)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()
    elif args.opt == 'noam':
        optimizer = chainer.optimizers.Adam(alpha=0, beta1=0.9, beta2=0.98, eps=1e-9)
    else:
        raise NotImplementedError('args.opt={}'.format(args.opt))

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    # Setup a converter
    converter = CustomConverter(subsampling_factor=model.subsample[0])

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # set up training iterator and updater
    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    accum_grad = args.accum_grad
    if ngpu <= 1:
        # make minibatch list (variable length)
        train = make_batchset(train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out, args.minibatches,
                              min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                              shortest_first=use_sortagrad,
                              count=args.batch_count,
                              batch_bins=args.batch_bins,
                              batch_frames_in=args.batch_frames_in,
                              batch_frames_out=args.batch_frames_out,
                              batch_frames_inout=args.batch_frames_inout)
        # hack to make batchsize argument as 1
        # actual batchsize is included in a list
        if args.n_iter_processes > 0:
            train_iters = [ToggleableShufflingMultiprocessIterator(
                TransformDataset(train, load_tr),
                batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
                shuffle=not use_sortagrad)]
        else:
            train_iters = [ToggleableShufflingSerialIterator(
                TransformDataset(train, load_tr),
                batch_size=1, shuffle=not use_sortagrad)]

        # set up updater
        updater = CustomUpdater(
            train_iters[0], optimizer, converter=converter, device=gpu_id, accum_grad=accum_grad)
    else:
        if args.batch_count not in ("auto", "seq") and args.batch_size == 0:
            raise NotImplementedError("--batch-count 'bin' and 'frame' are not implemented in chainer multi gpu")
        # set up minibatches
        train_subsets = []
        for gid in six.moves.xrange(ngpu):
            # make subset
            train_json_subset = {k: v for i, (k, v) in enumerate(train_json.items())
                                 if i % ngpu == gid}
            # make minibatch list (variable length)
            train_subsets += [make_batchset(train_json_subset, args.batch_size,
                                            args.maxlen_in, args.maxlen_out, args.minibatches)]

        # each subset must have same length for MultiprocessParallelUpdater
        maxlen = max([len(train_subset) for train_subset in train_subsets])
        for train_subset in train_subsets:
            if maxlen != len(train_subset):
                for i in six.moves.xrange(maxlen - len(train_subset)):
                    train_subset += [train_subset[i]]

        # hack to make batchsize argument as 1
        # actual batchsize is included in a list
        if args.n_iter_processes > 0:
            train_iters = [ToggleableShufflingMultiprocessIterator(
                TransformDataset(train_subsets[gid], load_tr),
                batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
                shuffle=not use_sortagrad)
                for gid in six.moves.xrange(ngpu)]
        else:
            train_iters = [ToggleableShufflingSerialIterator(
                TransformDataset(train_subsets[gid], load_tr),
                batch_size=1, shuffle=not use_sortagrad)
                for gid in six.moves.xrange(ngpu)]

        # set up updater
        updater = CustomParallelUpdater(
            train_iters, optimizer, converter=converter, devices=devices)

    # Set up a trainer
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler(train_iters),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))
    if args.opt == 'noam':
        from espnet.nets.chainer_backend.transformer.optimizer_rule import VaswaniRule
        trainer.extend(VaswaniRule('alpha', d=args.adim, warmup_steps=args.transformer_warmup_steps,
                                   scale=args.transformer_lr), trigger=(1, 'iteration'))
    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # set up validation iterator
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)

    if args.n_iter_processes > 0:
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        valid_iter, model, converter=converter, device=gpu_id))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        logging.info('Using custom PlotAttentionReport')
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=gpu_id)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(filename='snapshot.ep.{.updater.epoch}'), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    trainer.extend(extensions.snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode != 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best'),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode != 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best'),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best'),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').eps),
            trigger=(REPORT_INTERVAL, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    set_early_stop(trainer, args)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, att_reporter), trigger=(REPORT_INTERVAL, 'iteration'))

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    set_deterministic_chainer(args)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from ' + args.model)
    # To be compatible with v.0.3.0 models
    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.chainer_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    chainer_load(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(
            len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        chainer_load(args.rnnlm, rnnlm)
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        chainer_load(args.word_rnnlm, word_rnnlm)

        if rnnlm is not None:
            rnnlm = lm_chainer.ClassifierWithState(
                extlm_chainer.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_chainer.ClassifierWithState(
                extlm_chainer.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    # decode each utterance
    new_js = {}
    with chainer.no_backprop_mode():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            nbest_hyps = model.recognize(feat, args, train_args.char_list, rnnlm)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
