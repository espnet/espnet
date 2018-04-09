#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import logging
import os
import pickle

# chainer related
import chainer
from chainer import training
from chainer.training import extensions

# espnet related
from asr_chainer import ChainerSeqEvaluaterKaldi
from asr_chainer import ChainerSeqUpdaterKaldi
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import make_batchset
from asr_utils import restore_snapshot

# for e2e_mc
from beamformer import NB_MVDR
from e2e_asr_attctc import E2E
from e2e_asr_attctc import Loss
from e2e_asr_mc_attctc import E2E_MC

# for kaldi io
import kaldi_io_py
import lazy_io

# rnnlm
import lm_chainer

# numpy related
import matplotlib
import numpy as np
matplotlib.use('Agg')


def e2e_mc_converter_train(batch, readers):
    data = batch[0]
    utt_type = data[1]['utt_type']

    for data in batch:
        # noisy
        if utt_type == 'noisy':
            bidim = readers['noisy'][data[0].encode('ascii', 'ignore')].shape[1] / 2
            # separate real and imaginary part
            feat_real = readers['noisy'][data[0].encode('ascii', 'ignore')][:, :bidim]
            feat_imag = readers['noisy'][data[0].encode('ascii', 'ignore')][:, bidim:]
        # enhancement
        elif utt_type == 'enhan':
            bidim = readers['enhan'][0][data[0].encode('ascii', 'ignore')].shape[1] / 2
            # separate real and imaginary part
            feat_real = [reader[data[0].encode('ascii', 'ignore')][:, :bidim] for reader in readers['enhan']]
            feat_imag = [reader[data[0].encode('ascii', 'ignore')][:, bidim:] for reader in readers['enhan']]

        feat = {}
        feat['real'] = feat_real
        feat['imag'] = feat_imag

        data[1]['feat'] = feat

    return batch


def e2e_mc_converter_recog(name, readers):
    # enhancement
    bidim = readers[0][name].shape[1] / 2
    # separate real and imaginary part
    feat_real = [reader[name][:, :bidim] for reader in readers]
    feat_imag = [reader[name][:, bidim:] for reader in readers]

    feat = {}
    feat['real'] = feat_real
    feat['imag'] = feat_imag

    return feat


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
    with open(args.valid_label_noisy, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['idim'])
    odim = int(valid_json[utts[0]]['odim'])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # check attention type
    if args.atype not in ['noatt', 'dot', 'location']:
        raise NotImplementedError('chainer supports only noatt, dot, and location attention.')

    # get Mel-filterbank
    melmat = kaldi_io_py.read_mat(args.melmat)

    # get cmvn statistics
    stats = kaldi_io_py.read_mat(args.cmvn)
    dim = len(stats[0]) - 1
    count = stats[0][dim]
    cmvn_mean = stats[0][0:dim] / count
    cmvn_std = np.sqrt(stats[1][0:dim] / count - cmvn_mean * cmvn_mean)
    cmvn_stats = (cmvn_mean, cmvn_std)

    # set feature dims
    bidim = melmat.shape[1]
    eidim = melmat.shape[0]

    # specify model architecture
    enhan = NB_MVDR(bidim, args)
    asr = E2E(eidim, odim, args)
    e2e_mc = E2E_MC(enhan, asr, melmat, cmvn_stats)
    model = Loss(e2e_mc, args.mtlalpha)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((bidim, eidim, odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    gpu_id = int(args.gpu)
    logging.info('gpu id: ' + str(gpu_id))
    if gpu_id >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta(eps=args.eps)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    # read json data
    with open(args.train_label_noisy, 'rb') as f:
        train_json_noisy = json.load(f)['utts']
    with open(args.train_label_enhan, 'rb') as f:
        train_json_enhan = json.load(f)['utts']
    with open(args.valid_label_noisy, 'rb') as f:
        valid_json_noisy = json.load(f)['utts']
    with open(args.valid_label_enhan, 'rb') as f:
        valid_json_enhan = json.load(f)['utts']

    # add utt_type information for multi condition training
    for key, value in train_json_noisy.items():
        value['utt_type'] = 'noisy'
    for key, value in train_json_enhan.items():
        value['utt_type'] = 'enhan'
    for key, value in valid_json_noisy.items():
        value['utt_type'] = 'noisy'
    for key, value in valid_json_enhan.items():
        value['utt_type'] = 'enhan'

    # make minibatch list (variable length)
    train_noisy = make_batchset(train_json_noisy, args.batch_size,
                                args.maxlen_in, args.maxlen_out, args.minibatches)
    train_enhan = make_batchset(train_json_enhan, args.batch_size,
                                args.maxlen_in, args.maxlen_out, args.minibatches)
    valid_noisy = make_batchset(valid_json_noisy, args.batch_size,
                                args.maxlen_in, args.maxlen_out, args.minibatches)
    valid_enhan = make_batchset(valid_json_enhan, args.batch_size,
                                args.maxlen_in, args.maxlen_out, args.minibatches)

    # merge noisy and enhancement data
    train = train_noisy + train_enhan
    valid = valid_noisy + valid_enhan

    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # prepare Kaldi reader
    train_reader_noisy = lazy_io.read_dict_scp(args.train_feat_noisy)
    valid_reader_noisy = lazy_io.read_dict_scp(args.valid_feat_noisy)
    train_reader_enhan = [lazy_io.read_dict_scp(feat) for feat in args.train_feat_enhan]
    valid_reader_enhan = [lazy_io.read_dict_scp(feat) for feat in args.valid_feat_enhan]

    train_readers = {}
    train_readers['noisy'] = train_reader_noisy
    train_readers['enhan'] = train_reader_enhan

    valid_readers = {}
    valid_readers['noisy'] = valid_reader_noisy
    valid_readers['enhan'] = valid_reader_enhan

    # Set up a trainer
    updater = ChainerSeqUpdaterKaldi(
        train_iter, optimizer, train_readers, gpu_id, e2e_mc_converter_train)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(ChainerSeqEvaluaterKaldi(
        valid_iter, model, valid_readers, gpu_id, e2e_mc_converter_train))

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
        bidim, eidim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # get Mel-filterbank
    melmat = kaldi_io_py.read_mat(train_args.melmat)

    # get cmvn statistics
    stats = kaldi_io_py.read_mat(train_args.cmvn)
    dim = len(stats[0]) - 1
    count = stats[0][dim]
    cmvn_mean = stats[0][0:dim] / count
    cmvn_std = np.sqrt(stats[1][0:dim] / count - cmvn_mean * cmvn_mean)
    cmvn_stats = (cmvn_mean, cmvn_std)

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    enhan = NB_MVDR(bidim, train_args)
    asr = E2E(eidim, odim, train_args)
    e2e_mc = E2E_MC(enhan, asr, melmat, cmvn_stats)
    model = Loss(e2e_mc, train_args.mtlalpha)
    chainer.serializers.load_npz(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(len(train_args.char_list), 650))
        chainer.serializers.load_npz(args.rnnlm, rnnlm)
    else:
        rnnlm = None

    # prepare Kaldi reader
    names = [key for key, mat in lazy_io.read_mat_scp(args.recog_feat_enhan[0])]
    recog_reader = [lazy_io.read_dict_scp(feat) for feat in args.recog_feat_enhan]

    # read json data
    with open(args.recog_label_enhan, 'rb') as f:
        recog_json = json.load(f)['utts']

    new_json = {}
    for name in names:
        logging.info('decoding ' + name)
        feat = e2e_mc_converter_recog(name, recog_reader)
        if args.beam_size == 1:
            y_hat = e2e_mc.recognize(feat, args, train_args.char_list, rnnlm)
        else:
            nbest_hyps = e2e_mc.recognize(feat, args, train_args.char_list, rnnlm)
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
