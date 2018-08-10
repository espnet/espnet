#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import collections
import json
import logging
import math
import multiprocessing
import os
import pickle
import six
import sys

# chainer related
import chainer

from chainer import cuda
from chainer import reporter as reporter_module
from chainer import training
from chainer import Variable

from chainer.training import extensions
from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads
from chainer.training.updaters.multiprocess_parallel_updater import scatter_params

# espnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import load_labeldict
from asr_utils import make_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from e2e_asr import E2E
from e2e_asr import Loss

# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_chainer
import lm_chainer

# numpy related
import matplotlib
import numpy as np
matplotlib.use('Agg')


# copied from https://github.com/chainer/chainer/blob/master/chainer/optimizer.py
def sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            if x is not None:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for chainer'''

    def __init__(self, train_iter, optimizer, converter, device):
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=device)

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
        x = self.converter(batch)

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(*x)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in optimizer.target.params(False)]))
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.update()


class CustomParallelUpdater(training.updaters.MultiprocessParallelUpdater):
    '''Custom parallel updater for chainer'''

    def __init__(self, train_iters, optimizer, converter, devices):
        super(CustomParallelUpdater, self).__init__(
            train_iters, optimizer, converter=converter, devices=devices)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            from cupy.cuda import nccl
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            x = self.converter(batch)

            loss = self._master(*x)

            self._master.cleargrads()
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

            # check gradient value
            grad_norm = np.sqrt(sum_sqnorm(
                [p.grad for p in optimizer.target.params(False)]))
            logging.info('grad norm={}'.format(grad_norm))

            # update
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.update()

            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT,
                                0, null_stream.ptr)

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
            from cupy.cuda import nccl
            self._master.to_gpu(self._devices[0])
            if len(self._devices) > 1:
                comm_id = nccl.get_unique_id()
                self._send_message(("set comm_id", comm_id))
                self.comm = nccl.NcclCommunicator(len(self._devices),
                                                  comm_id, 0)


class CustomWorker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(CustomWorker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.model = master._master
        self.converter = master.converter
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)

    def setup(self):
        from cupy.cuda import nccl
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)

        self.model.to_gpu(self.device)
        self.reporter = reporter_module.Reporter()
        self.reporter.add_observer('main', self.model)

    def run(self):
        from cupy.cuda import nccl
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
                x = self.converter(batch)
                observation = {}
                with self.reporter.scope(observation):
                    loss = self.model(*x)

                self.model.cleargrads()
                loss.backward()
                loss.unchain_backward()

                del loss

                gg = gather_grads(self.model)
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl.NCCL_FLOAT,
                                 nccl.NCCL_SUM, 0,
                                 null_stream.ptr)
                del gg
                self.model.cleargrads()
                gp = gather_params(self.model)
                self.comm.bcast(gp.data.ptr, gp.size,
                                nccl.NCCL_FLOAT, 0,
                                null_stream.ptr)
                scatter_params(self.model, gp)
                gp = None


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, device, subsamping_factor=1):
        self.device = device
        self.subsamping_factor = subsamping_factor
        self.xp = cuda.cupy if device != -1 else np

    def __call__(self, batch, *args):
        # batch should be located in list
        assert len(batch) == 1
        batch = batch[0]

        # load acoustic features and target sequence of token ids
        xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]
        tids = [b[1]['output'][0]['tokenid'].split() for b in batch]

        # remove 0-output-length utterances
        filtered_index = filter(lambda i: len(tids[i]) > 0, range(len(xs)))
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
        if len(sorted_index) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(sorted_index)))
        xs = [xs[i] for i in sorted_index]
        ys = [np.fromiter(map(int, tids[i]), dtype=np.int64) for i in sorted_index]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = [x.shape[0] for x in xs]

        # convert to Variable
        xs = [Variable(self.xp.array(x, dtype=np.float32)) for x in xs]
        ilens = self.xp.array(ilens, dtype=np.int32)
        ys = [Variable(self.xp.array(y, dtype=np.int32)) for y in ys]

        return xs, ilens, ys


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
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # set up training iterator and updater
    converter = CustomConverter(gpu_id, e2e.subsample[0])
    if ngpu <= 1:
        # make minibatch list (variable length)
        train = make_batchset(train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out, args.minibatches)
        # hack to make batchsize argument as 1
        # actual batchsize is included in a list
        train_iter = chainer.iterators.SerialIterator(train, 1)

        # set up updater
        updater = CustomUpdater(
            train_iter, optimizer, converter=converter, device=gpu_id)
    else:
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
        updater = CustomParallelUpdater(
            train_iters, optimizer, converter=converter, devices=devices)

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
    trainer.extend(extensions.Evaluator(
        valid_iter, model, converter=converter, device=gpu_id))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.predictor.calculate_all_attentions
        else:
            att_vis_fn = model.predictor.calculate_all_attentions
        trainer.extend(PlotAttentionReport(att_vis_fn, data, args.outdir + "/att_ws", converter),
                       trigger=(1, 'epoch'))

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
    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best'),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
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

    if args.word_rnnlm:
        if not args.word_dict:
            logging.error('word dictionary file is not specified for the word RNNLM.')
            sys.exit(1)

        word_dict = load_labeldict(args.word_dict)
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(len(word_dict), 650))
        chainer.serializers.load_npz(args.word_rnnlm, word_rnnlm)

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
        recog_json = json.load(f)['utts']

    new_json = {}
    for name in recog_json.keys():
        feat = kaldi_io_py.read_mat(recog_json[name]['input'][0]['feat'])
        logging.info('decoding ' + name)
        nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
        # get 1best and remove sos
        y_hat = nbest_hyps[0]['yseq'][1:]
        y_true = map(int, recog_json[name]['output'][0]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = dict()
        new_json[name]['utt2spk'] = recog_json[name]['utt2spk']

        # add 1-best recognition results to json
        logging.debug("dump token id")
        out_dic = dict()
        for _key in recog_json[name]['output'][0]:
            out_dic[_key] = recog_json[name]['output'][0][_key]

        # TODO(karita) make consistent to chainer as idx[0] not idx
        out_dic['rec_tokenid'] = " ".join(
            [str(idx[0]) for idx in y_hat])
        logging.debug("dump token")
        out_dic['rec_token'] = " ".join(seq_hat)
        logging.debug("dump text")
        out_dic['rec_text'] = seq_hat_text

        new_json[name]['output'] = [out_dic]
        # TODO(nelson): Modify this part when saving more than 1 hyp is enabled
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
