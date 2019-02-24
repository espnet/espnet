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
from espnet.asr.asr_utils import freeze_parameters
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import load_inputs_and_targets
from espnet.asr.asr_utils import make_batchset
from espnet.asr.asr_utils import merge_batchsets
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.nets.e2e_asr_cyc_th import E2E
from espnet.nets.e2e_asr_cyc_th import Loss
from espnet.nets.e2e_asr_cyc_th import pad_list

from espnet.nets.e2e_asr_cyc_th import ExpectedLoss
from espnet.nets.e2e_tts_cyc_th import ASR2Tacotron2Loss
from espnet.nets.e2e_tts_cyc_th import TacotronTTS2ASRLoss
from espnet.nets.e2e_tts_cyc_th import Tacotron2ASRLoss
from espnet.nets.e2e_tts_cyc_th import Tacotron2TTELoss

from espnet.asr.asr_pytorch import CustomConverter as ASRConverter
from espnet.tts.tts_cyc_pytorch import CustomConverter as TTSConverter

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


class CustomEvaluator(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, tts2asr_model, asr_model, tts_model,
                 grad_clip_threshold, train_iter,
                 optimizer_B, asr_converter, tts_converter, device, ngpu,
                 update_asr_only=False, freeze_asr=False, freeze_tts=False):
        super(CustomUpdater, self).__init__(train_iter, optimizer_B)

        self.tts2asr_model = tts2asr_model
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.model = asr_model
        self.grad_clip_threshold = grad_clip_threshold
        self.asr_converter = asr_converter
        self.tts_converter = tts_converter
        self.device = device
        self.ngpu = ngpu
        self.update_asr_only = update_asr_only
        self.freeze_asr = freeze_asr
        self.freeze_tts = freeze_tts

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer_B = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        if np.array_equal(batch[0][0][0], batch[0][1][0]):
            if isinstance(batch[0][0][0][0], np.int64):
                xs, ilens, ys, labels, olens, spembs, spcs = self.tts_converter(batch, self.device)
                self.asr_model.train()
                self.tts_model.train()
                loss_B = 1. / self.ngpu * self.tts2asr_model(xs, ilens, ys, labels, olens=olens)
                optimizer_B.zero_grad()  # Clear the parameter gradients
                if self.ngpu > 1:
                    loss_B.sum().backward()
                else:
                    loss_B.backward()
                loss_B.detach()  # Truncate the graph
                # compute the gradient norm to check if it is normal or not
                grad_norm_B = torch.nn.utils.clip_grad_norm_(
                    self.tts2asr_model.parameters(), self.grad_clip_threshold)
                logging.info('grad norm B={}'.format(grad_norm_B))
                if math.isnan(grad_norm_B):
                    logging.warning('grad norm is nan. Do not update model.')
                else:
                    optimizer_B.step()


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1

    def transform(self, item):
        return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad

def raw_merge_batchsets(train_subset_A, train_subset_B, shuffle=True):
    ndata_A = len(train_subset_A)
    ndata_B = len(train_subset_B)
    ndata=min(ndata_A, ndata_B)
    train = []
    for num, data in enumerate(zip(train_subset_A[:ndata], train_subset_B[:ndata])):
        train.extend(data)
    if shuffle:
        import random
        random.shuffle(train)
    return train


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
        if 'conf' in args.asr_model_conf:
            with open(args.asr_model_conf, "rb") as f:
                logging.info('reading a model config file from' + args.asr_model_conf)
                import pickle
                idim, odim, train_args = pickle.load(f)
        elif 'json' in args.asr_model_conf:
            idim, odim, train_args = get_model_conf(args.asr_model,
                                                    conf_path=args.asr_model_conf)
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
        torch_load(args.rnnlm, rnnlm)
        e2e.rnnlm = rnnlm
    else:
        rnnlm = None

    if args.freeze == "encattdec":
        asr_model, size = freeze_parameters(asr_model, 16)
        logging.info("no of parameters frozen are: " + str(size))
    elif args.freeze == "dec":
        asr_model, size = freeze_parameters(asr_model, 0, "att")
        logging.info("no of parameters frozen are: " + str(size))
    elif args.freeze == "attdec":
        asr_model, size = freeze_parameters(asr_model, 0)
        logging.info("no of parameters frozen are: " + str(size))
    elif args.freeze == "encatt":
        asr_model, size = freeze_parameters(asr_model, 16, "dec")
        logging.info("no of parameters frozen are: " + str(size))
    elif args.freeze == "enc":
        asr_model, size = freeze_parameters(asr_model, 16, "att", "dec")
        logging.info("no of parameters frozen are: " + str(size))
    else:
        logging.warn("Only Disney is frozen")

    # setup loss functions
    if args.expected_loss:
        # need to specify a loss function (loss_fn) to compute the expected
        # loss
        if args.expected_loss == 'tts':
            from taco_cycle_consistency import load_tacotron_loss
            #assert args.tts_model, "Need to provide --tts-model and set --expected-loss tts"
            loss_fn = load_tacotron_loss(args.tts_model_conf, args.tts_model,
                                         asr_model.reporter, args)
        elif args.expected_loss == 'none':
            loss_fn = None
        else:
            raise NotImplemented(
                'Unknown expected loss: %s' % args.expected_loss
            )
        if args.check:
            asr2tts_model = ASR2Tacotron2Loss(loss_fn.model, asr_model.predictor,
                                              reporter=asr_model.reporter,
                                              weight=args.teacher_weight,
                                              rnnlm=rnnlm)
        else:
            asr2tts_model = ExpectedLoss(asr_model.predictor, args, loss_fn=loss_fn,
                                     reporter=asr_model.reporter,
                                     rnnlm=rnnlm, lm_loss_weight=args.lm_loss_weight)

        if loss_fn is not None:
            if args.generator == 'tte':
                tts_model = Tacotron2TTELoss(loss_fn.model, asr_model.predictor,
                                             reporter=asr_model.reporter)
                tts2asr_model = Tacotron2ASRLoss(loss_fn.model, asr_model.predictor,
                                             reporter=asr_model.reporter,
                                             weight=args.teacher_weight)

            elif args.generator == 'tts':
                tts_model = loss_fn.model

                tts2asr_model = TacotronTTS2ASRLoss(loss_fn.model, asr_model.predictor,
                                             reporter=asr_model.reporter,
                                             weight=args.teacher_weight)
        else:
            tts2asr_model = asr_model
            tts_model = None
    else:
        logging.warn("using asr as asr2tts model")
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

    reporter = asr_model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        tts2asr_model = torch.nn.DataParallel(tts2asr_model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    tts2asr_model = tts2asr_model.to(device)
    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer_B = torch.optim.Adadelta(
            tts2asr_model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer_B = torch.optim.Adam(tts2asr_model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer_B, "target", reporter)
    setattr(optimizer_B, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(e2e.subsample[0])

    # read json data
    train_json = []
    for idx in range(len(args.train_json)):
        with open(args.train_json[idx], 'rb') as f:
            train_json.append(json.load(f)['utts'])
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json[1], args.batch_size, args.maxlen_in,
                                    args.maxlen_out, args.minibatches,
                                    min_batch_size=args.ngpu if args.ngpu > 1 else 1)

    #train = raw_merge_batchsets(train_subsets_A, train_subsets_B, shuffle=False)
    valid = make_batchset(valid_json, args.batch_size, args.maxlen_in, args.maxlen_out,
                          args.minibatches, min_batch_size=args.ngpu if args.ngpu > 1 else 1)

    # hack to make batchsze argument as 1
    # actual batchsize is included in a list
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
        tts2asr_model,
        asr_model,
        tts_model,
        args.grad_clip,
        train_iter,
        optimizer_B,
        ASRConverter(e2e.subsample[0]),
        TTSConverter(),
        device,
        args.ngpu,
        update_asr_only=args.update_asr_only,
        freeze_asr=args.freeze_asr,
        freeze_tts=args.freeze_tts)
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
    trainer.extend(extensions.snapshot_object(tts2asr_model, 'model_B.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(asr_model, 'model.acc.asr.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))
        trainer.extend(extensions.snapshot_object(tts2asr_model, 'model_B.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
            trainer.extend(restore_snapshot(tts2asr_model, args.outdir + '/model_B.acc.best', load_fn=torch_load),
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
                   'main/acc', 'validation/main/loss', 'validation/main/acc', 'elapsed_time']
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


def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    torch_load(args.model, model)
    e2e.recog_args = args
    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # gpu
    if args.ngpu == 1:
        gpu_id = range(args.ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
                nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)
    else:
        try:
            from itertools import zip_longest as zip_longest
        except Exception:
            from itertools import izip_longest as zip_longest

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data
        keys = list(js.keys())
        feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
        sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
        keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                feats = [kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
                         for name in names]
                nbest_hyps = e2e.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)
                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
