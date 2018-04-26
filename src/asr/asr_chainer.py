#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import collections
import copy
import json
import logging
import math
import multiprocessing
import os
import pickle
import six

# chainer related
import chainer

from chainer import cuda
from chainer import function
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads
from chainer.training.updaters.multiprocess_parallel_updater import scatter_params

# espnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import converter_kaldi
from asr_utils import delete_feat
from asr_utils import make_batchset
from asr_utils import restore_snapshot
from e2e_asr_attctc import E2E
from e2e_asr_attctc import Loss

# for kaldi io
import kaldi_io_py
import lazy_io

# rnnlm
import lm_chainer

# numpy related
import matplotlib
import numpy as np
matplotlib.use('Agg')


class ChainerSeqEvaluaterKaldi(extensions.Evaluator):
    '''Custom evaluater with Kaldi reader for chainer'''

    def __init__(self, iterator, target, reader, device):
        super(ChainerSeqEvaluaterKaldi, self).__init__(
            iterator, target, device=device)
        self.reader = reader

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        '''evaluate over iterator'''
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        # for multi gpu calculation
        chainer.cuda.get_device_from_id(self.device).use()
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                # read scp files
                # x: original json with loaded features
                #    will be converted to chainer variable later
                # batch only has one minibatch utterance, which is specified by batch[0]
                x = converter_kaldi(batch[0], self.reader)
                with function.no_backprop_mode():
                    eval_func(x)
                    delete_feat(x)

            summary.add(observation)

        return summary.compute_mean()


class ChainerSeqUpdaterKaldi(training.StandardUpdater):
    '''Custom updater with Kaldi reader for chainer'''

    def __init__(self, train_iter, optimizer, reader, device):
        super(ChainerSeqUpdaterKaldi, self).__init__(
            train_iter, optimizer, device=device)
        self.reader = reader

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
        x = converter_kaldi(batch[0], self.reader)

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(x)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = np.sqrt(self._sum_sqnorm(
            [p.grad for p in optimizer.target.params(False)]))
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.update()
        delete_feat(x)

    # copied from https://github.com/chainer/chainer/blob/master/chainer/optimizer.py
    def _sum_sqnorm(self, arr):
        sq_sum = collections.defaultdict(float)
        for x in arr:
            with cuda.get_device_from_array(x) as dev:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
        return sum([float(i) for i in six.itervalues(sq_sum)])


class ChainerMultiProcessParallelUpdaterKaldi(training.updaters.MultiprocessParallelUpdater):
    '''Custom parallel updater with Kaldi reader for chainer'''

    def __init__(self, train_iters, optimizer, reader, devices):
        super(ChainerMultiProcessParallelUpdaterKaldi, self).__init__(
            train_iters, optimizer, devices=devices)
        self.reader = reader

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            x = converter_kaldi(batch[0], self.reader)

            loss = self._master(x)

            self._master.cleargrads()
            loss.backward()
            loss.unchain_backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 cupy.cuda.nccl.NCCL_FLOAT,
                                 cupy.cuda.nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg

            # check gradient value
            grad_norm = np.sqrt(self._sum_sqnorm(
                [p.grad for p in optimizer.target.params(False)]))
            logging.info('grad norm={}'.format(grad_norm))

            # update
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.update()

            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, cupy.cuda.nccl.NCCL_FLOAT,
                                0, null_stream.ptr)

            delete_feat(x)

    def setup_workers(self):
        if self._initialized:
            return
        self._initialized = True

        self._master.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = CustomWorker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self._devices[0]):
            self._master.to_gpu(self._devices[0])
            if len(self._devices) > 1:
                comm_id = cupy.cuda.nccl.get_unique_id()
                self._send_message(("set comm_id", comm_id))
                self.comm = cupy.cuda.nccl.NcclCommunicator(len(self._devices),
                                                            comm_id, 0)

    # copied from https://github.com/chainer/chainer/blob/master/chainer/optimizer.py
    def _sum_sqnorm(self, arr):
        sq_sum = collections.defaultdict(float)
        for x in arr:
            with cuda.get_device_from_array(x) as dev:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
        return sum([float(i) for i in six.itervalues(sq_sum)])


class CustomWorker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(CustomWorker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.model = master._master
        self.reader = master.reader
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)

    def setup(self):
        _, comm_id = self.pipe.recv()
        self.comm = cupy.cuda.nccl.NcclCommunicator(self.n_devices, comm_id,
                                                    self.proc_id)

        self.model.to_gpu(self.device)
        self.reporter = reporter_module.Reporter()
        self.reporter.add_observer('main', self.model)

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()
        gp = None
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                break
            if job == 'update':
                # For reducing memory
                self.model.cleargrads()

                batch = self.iterator.next()
                x = converter_kaldi(batch[0], self.reader)
                observation = {}
                with self.reporter.scope(observation):
                    loss = self.model(x)

                self.model.cleargrads()
                loss.backward()
                loss.unchain_backward()

                del loss

                gg = gather_grads(self.model)
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 cupy.cuda.nccl.NCCL_FLOAT,
                                 cupy.cuda.nccl.NCCL_SUM, 0,
                                 null_stream.ptr)
                del gg
                self.model.cleargrads()
                gp = gather_params(self.model)
                self.comm.bcast(gp.data.ptr, gp.size,
                                cupy.cuda.nccl.NCCL_FLOAT, 0,
                                null_stream.ptr)
                scatter_params(self.model, gp)
                gp = None

                delete_feat(x)


def train(args):
    '''Run training'''
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    # seed setting (chainer seed may not need it)
    os.environ['CHAINER_SEED'] = str(args.seed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('chainer type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        chainer.config.cudnn_deterministic = False
        logging.info('chainer cudnn deterministic is disabled')
    else:
        chainer.config.cudnn_deterministic = True

    # check cuda and cudnn availability
    if not chainer.cuda.available:
        logging.warning('cuda is not available')
    if not chainer.cuda.cudnn_enabled:
        logging.warning('cudnn is not available')

    # get input and output dimension info
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['idim'])
    odim = int(valid_json[utts[0]]['odim'])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # check attention type
    if args.atype not in ['noatt', 'dot', 'location']:
        raise NotImplementedError('chainer supports only noatt, dot, and location attention.')

    # specify model architecture
    e2e = E2E(idim, odim, args)
    model = Loss(e2e, args.mtlalpha)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((idim, odim, args), f)
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
        args.batch_size = math.ceil(args.batch_size / ngpu)
        devices = {'main': gpu_id}
        for gid in six.moves.xrange(1, ngpu):
            devices['sub_%d' % gid] = gid
        logging.info('multi gpu calculation (#gpus = %d).' % ngpu)
    else:
        gpu_id = -1
        logging.info('cpu calculation')

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta(eps=args.eps)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    # read json data
    with open(args.train_label, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']

    # prepare Kaldi reader
    train_reader = lazy_io.read_dict_scp(args.train_feat)
    valid_reader = lazy_io.read_dict_scp(args.valid_feat)

    # set up training iterator and updater
    if ngpu <= 1:
        # make minibatch list (variable length)
        train = make_batchset(train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out, args.minibatches)
        # hack to make batchsize argument as 1
        # actual batchsize is included in a list
        train_iter = chainer.iterators.SerialIterator(train, 1)

        # set up updater
        updater = ChainerSeqUpdaterKaldi(
            train_iter, optimizer, train_reader, gpu_id)
    else:
        # import cupy only when multiple GPUs
        import cupy
        # set up minibatches
        train_subsets = []
        for gid in six.moves.xrange(ngpu):
            # make subset
            train_json_subset = {k: v for i, (k, v) in enumerate(train_json.viewitems())
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
        train_iters = [chainer.iterators.MultiprocessIterator(
            train_subsets[gid], 1, n_processes=1)
            for gid in six.moves.xrange(ngpu)]

        # set up updater
        updater = ChainerMultiProcessParallelUpdaterKaldi(
            train_iters, optimizer, train_reader, devices)

    # Set up a trainer
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # set up validation iterator
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(ChainerSeqEvaluaterKaldi(
        valid_iter, model, valid_reader, device=gpu_id))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

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
    trainer.extend(extensions.snapshot_object(model, 'model.acc.best'),
                   trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
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
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').eps),
            trigger=(100, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(100, 'iteration'))

    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''
    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    # seed setting (chainer seed may not need it)
    os.environ["CHAINER_SEED"] = str(args.seed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    chainer.serializers.load_npz(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(len(train_args.char_list), 650))
        chainer.serializers.load_npz(args.rnnlm, rnnlm)
    else:
        rnnlm = None

    # prepare Kaldi reader
    reader = kaldi_io_py.read_mat_ark(args.recog_feat)

    # read json data
    with open(args.recog_label, 'rb') as f:
        recog_json = json.load(f)['utts']

    new_json = {}
    for name, feat in reader:
        logging.info('decoding ' + name)
        if args.beam_size == 1:
            y_hat = e2e.recognize(feat, args, train_args.char_list, rnnlm)
        else:
            nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
            # get 1best and remove sos
            y_hat = nbest_hyps[0]['yseq'][1:]
        y_true = map(int, recog_json[name]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = recog_json[name]

        # add 1-best recognition results to json
        new_json[name]['rec_tokenid'] = " ".join(
            [str(idx[0]) for idx in y_hat])
        new_json[name]['rec_token'] = " ".join(seq_hat)
        new_json[name]['rec_text'] = seq_hat_text

        # add n-best recognition results with scores
        if args.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] \
                    = " ".join([str(idx[0]) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = hyp['score']

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))
