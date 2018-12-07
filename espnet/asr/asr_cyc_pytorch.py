#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os

# chainer related
import chainer

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch

# espnet related
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import load_inputs_and_targets
from espnet.asr.asr_utils import make_batchset
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.nets.e2e_asr_th import E2E
from espnet.nets.e2e_asr_th import Loss
from espnet.nets.e2e_asr_th import pad_list

from espnet.nets.e2e_asr_th import ExpectedLoss
from espnet.nets.e2e_tts_th import Tacotron2Loss
from espnet.nets.e2e_tts_th import Tacotron2ASRLoss
from espnet.nets.e2e_tts_th import Tacotron2TTELoss

from espnet.asr.asr_pytorch import CustomEvaluator
from espnet.asr.asr_pytorch import CustomConverter as ASRConverter
from espnet.tts.tts_pytorch import CustomConverter as TTSConverter

# for kaldi io
import kaldi_io_py

# rnnlm
import espnet.lm.extlm_pytorch as extlm_pytorch
import espnet.lm.lm_pytorch as lm_pytorch

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

REPORT_INTERVAL = 100


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, asr2tts_model, tts2asr_model, asr_model, tts_model,
                 grad_clip_threshold, train_iter,
                 optimizer, asr_converter, tts_converter, device, ngpu
                update_asr_only=False, freeze_asr=False, freeze_tts=False):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.asr2tts_model = asr2tts_model
        self.tts2asr_model = tts2asr_model
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.grad_clip_threshold = grad_clip_threshold
        self.asr_converter = asr_converter
        self.tts_converter = tts_converter
        self.device = device
        self.ngpu = ngpu
        self.update_asr_only = update_asr_only
        self.freeze_asr = freeze_asr
        self.freeze_tts = freeze_tts

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        utt_type = batch[0][0][1]['utt_type'] if 'utt_type' in batch[0][0][1] else 0
        if utt_type == 0: # paired data training
            if self.update_asr_only:
                models = [ self.asr_model ]
            else:
                models = [ self.asr_model, self.tts_model ]
        elif utt_type == 1: # ASR with TTS loss
            models = [ self.asr2tts_model ]
        elif utt_type == 2: # TTS with ASR loss
            models = [ self.tts2asr_model ]
        else:
            raise NotImplementedError

        for n, model in enumerate(models):
            # Compute the loss at this time step and accumulate it
            if utt_type==0 and n==0:  # asr with CE loss
                self.asr_model.train()
                x = self.asr_converter(batch, self.device)
                loss = 1. / self.num_gpu * model(*x)
            elif utt_type==0 and n==1:  # tts with MSE loss
                self.asr_model.eval()
                self.tts_model.train()
                xs, ilens, ys, labels, olens, spembs = self.tts_converter(batch, self.device)
                loss = 1. / self.num_gpu * model(xs, ilens, ys, labels, olens=olens)
            elif utt_type==1:  # asr with tts loss
                self.asr_model.train()
                x = self.asr_converter(batch, self.device)
                loss = 1. / self.num_gpu * model(*x)
            else: #utt_type==2  tts with asr loss
                self.asr_model.train()
                self.tts_model.train()
                xs, ilens, ys, labels, olens, spembs = self.tts_converter(batch, self.device)
                loss = 1. / self.num_gpu * model(xs, ilens)

            optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()  # Backprop
            loss.detach()  # Truncate the graph
            if utt_type == 2:
                if self.freeze_asr:
                    self.asr_model.zero_grad()
                if self.freeze_tts:
                    self.tts_model.zero_grad()
            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.grad_clip_threshold)
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
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

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
    if args.asr_model_conf:
        with open(args.asr_model_conf, "rb") as f:
            logging.info('reading a model config file from' + args.asr_model_conf)
            idim, odim, train_args = pickle.load(f)
        e2e = E2E(idim, odim, train_args)
    else:
        e2e = E2E(idim, odim, args)
    asr_model = Loss(e2e, args.mtlalpha, weight=args.teacher_weight)
    if args.asr_model:
        asr_model.load_state_dict(torch.load(args.asr_model))

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch.load(args.rnnlm, rnnlm)
        e2e.rnnlm = rnnlm

    # setup loss functions
    if args.expected_loss:
        # need to specify a loss function (loss_fn) to compute the expected
        # loss
        if args.expected_loss == 'tts':
            from taco_cycle_consistency import (
                load_tacotron_loss,
                sanity_check_json
            )
            assert args.tts_model, \
                "Need to provide --tts-model and set --expected-loss tts"
            #sanity_check_json(valid_json)
            loss_fn = load_tacotron_loss(args.tts_model_conf, args.tts_model)
        elif args.expected_loss == 'none':
            loss_fn = None
        else:
            raise NotImplemented(
                'Unknown expected loss: %s' % args.expected_loss
            )
        asr2tts_model = ExpectedLoss(asr_model.predictor, args, loss_fn=loss_fn,
                                     reporter=asr_model.reporter,
                                     rnnlm=rnnlm, lm_loss_weight=args.lm_loss_weight)
        if loss_fn is not None:
            tts_model = Tacotron2TTELoss(loss_fn.model, asr_model.predictor, reporter=asr_model.reporter)
            tts2asr_model = Tacotron2ASRLoss(loss_fn.model, asr_model.predictor, reporter=asr_model.reporter, weight=args.teacher_weight)
        else:
            tts2asr_model = asr_model
            tts_model = None
    else:
        asr2tts_model = asr_model
        tts2asr_model = asr_model
        tts_model = None

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        asr2tts_model = torch.nn.DataParallel(asr2tts_model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    asr2tts_model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            tts2asr_model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(tts2asr_model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))


    # read json data
    train_json = []
    for n in range(len(args.train_json)):
        with open(args.train_json[n], 'rb') as f:
            train_json.append(json.load(f)['utts'])
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train_subsets = []
    for tj in train_json:
        train_subsets.append(make_batchset(tj, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1))
    train = merge_batchsets(train_subsets, shuffle=True)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = chainer.iterators.SerialIterator(
            TransformDataset(train, converter.transform),
            batch_size=1)
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        asr2tts_model,
        tts2asr_model,
        asr_model,
        tts_model,
        args.grad_clip, 
        train_iter, 
        optimizer, 
        ASRConverter(e2e.subsample[0])
        TTSConverter(gpu_id),
        device, 
        args.ngpu,
        update_asr_only=args.update_asr_only,
        freeze_asr=args.freeze_asr,
        freeze_tts=args.freeze_tts
    )
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(asr_model, valid_iter, reporter, converter, device))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(asr_model, "module"):
            att_vis_fn = asr_model.module.predictor.calculate_all_attentions
        else:
            att_vis_fn = asr_model.predictor.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, device=device), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'main/loss_cyc', 'main/loss_att',
                                          'main/loss_l1', 'main/loss_mse', 'main/loss_bce',
                                         'validation/main/loss'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    trainer.extend(extensions.snapshot_object(asr2tts_model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(asr2tts_model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
            trainer.extend(restore_snapshot(asr2tts_model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(asr2tts_model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_att', 'main/loss_cyc',
                   'main/loss_l1', 'main/loss_mse', 'main/loss_bce',
                   'main/acc', 'validation/main/loss', 'validation/main/acc',
                   'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(REPORT_INTERVAL, 'iteration'))
        report_keys.append('eps')
    if args.report_cer:
        report_keys.append('validation/main/cer')
    if args.report_wer:
        report_keys.append('validation/main/wer')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    # Run the training
    trainer.run()

