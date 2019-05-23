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
from espnet.asr.asr_utils import make_batchset
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.asrtts_utils import freeze_parameters
from espnet.asr.asrtts_utils import load_inputs_and_targets
from espnet.asr.asrtts_utils import load_inputs_spk_and_targets
from espnet.asr.asrtts_utils import merge_batchsets
from espnet.asr.asrtts_utils import remove_output_layer
from espnet.nets.pytorch_backend.e2e_asr import E2E
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.e2e_asrtts import E2E as asrtts
from espnet.utils.io_asrttsutils import LoadInputsAndTargetsASRTTS
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator

from espnet.nets.pytorch_backend.e2e_asrtts import Tacotron2ASRLoss

from espnet.asr.pytorch_backend.asr import CustomConverter as ASRConverter

# for kaldi io
import kaldi_io_py

# rnnlm
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
import espnet.lm.pytorch_backend.lm as lm_pytorch

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

    def __init__(self, asr2tts_model, tts2asr_model, asr_model, tts_model,
                 grad_clip_threshold, train_iter,
                 optimizer, asr_converter, asrtts_converter,
                 device, ngpu,
                 alpha=0.5, zero_att=False):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.asr2tts_model = asr2tts_model
        self.tts2asr_model = tts2asr_model
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.grad_clip_threshold = grad_clip_threshold
        self.asr_converter = asr_converter
        self.asrtts_converter = asrtts_converter
        self.device = device
        self.ngpu = ngpu
        self.alpha = alpha
        self.zero_att = zero_att

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        # Get the next batch ( a list of json files)
        loss = 0.0
        batch = train_iter.next()
        if len(batch[0]) == 3:
            utt_type = 'unpaired'
        elif len(batch[0]) == 2:
            utt_type = 'paired'
        else:
            raise NotImplementedError
        if utt_type == 'unpaired':
            # Compute the loss at this time step and accumulate it
            xs_asr, ilens_asr, ys_asr, xs_tts, ys_tts, ilens_tts, labels, \
                olens_tts, spembs = self.asrtts_converter(batch, self.device)
            self.asr_model.train()
            self.tts_model.train()
            logging.info("unpaired speech training")
            x = (xs_asr, ilens_asr, ys_asr, spembs)
            loss_asr2tts = self.asr2tts_model(*x)
            logging.info("unpaired text training")
            self.tts_model.train()
            # self.tts_model.eval()
            self.asr_model.train()
            loss_tts2asr = self.tts2asr_model(xs_tts, ilens_tts, ys_tts, labels,
                                              olens_tts, spembs)  # zero_att=self.zero_att)
            loss = self.alpha * loss_asr2tts + (1 - self.alpha) * loss_tts2asr
        else:
            self.asr_model.train()
            x = self.asr_converter(batch, self.device)
            logging.info("paired data training")
            loss_att, _, _, _, _ = self.asr_model(*x)
            loss = 1. / self.ngpu * loss_att

        optimizer.zero_grad()
        if self.ngpu > 1:
            loss.sum().backward()
        else:
            loss.backward()
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(self.asr2tts_model.parameters(),
                                                   self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()


class CustomASRTTSConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, use_speaker_embedding=False):
        self.use_speaker_embedding = use_speaker_embedding

    def transform(self, item):
        if self.use_speaker_embedding:
            return load_inputs_spk_and_targets(item,
                                               use_speaker_embedding=self.use_speaker_embedding)
        else:
            return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        if self.use_speaker_embedding:
            xs, ys, spembs = batch[0]
        else:
            xs, ys = batch[0]

        # get batch of lengths of input sequences
        ilens_asr = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        ilens_tts = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)
        olens_tts = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        ilens_tts = sorted(ilens_tts, reverse=True)
        # perform padding and convert to tensor
        xs_asr = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ys_asr = pad_list([torch.from_numpy(y).long() for y in ys], -1).to(device)
        xs_tts = pad_list([torch.from_numpy(y).long() for y in ys], 0).to(device)
        ys_tts = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        labels = ys_tts.new_zeros(ys_tts.size(0), ys_tts.size(1))
        for i, l in enumerate(olens_tts):
            labels[i, l - 1:] = 1.0

        if self.use_speaker_embedding:
            spembs = torch.from_numpy(np.array(spembs)).float().to(device)
        else:
            spembs = None
        return xs_asr, ilens_asr, ys_asr, xs_tts, ys_tts, ilens_tts, labels, olens_tts, spembs


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1, use_speaker_embedding=False):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1
        self.use_speaker_embedding = use_speaker_embedding

    def transform(self, item):
        if self.use_speaker_embedding:
            return load_inputs_spk_and_targets(item,
                                               use_speaker_embedding=self.use_speaker_embedding)
        else:
            return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        if self.use_speaker_embedding:
            try:
                xs, ys, spembs = batch[0]
            except ValueError:
                xs, ys = batch[0]
        else:
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

        if self.use_speaker_embedding:
            try:
                spembs = torch.from_numpy(np.array(spembs)).float().to(device)
                return xs_pad, ilens, ys_pad, spembs
            except UnboundLocalError:
                return xs_pad, ilens, ys_pad
        else:
            return xs_pad, ilens, ys_pad


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
        asr_model = asrtts(idim, odim, train_args)
    else:
        asr_model = asrtts(idim, odim, args)
    if args.asr_model:
        if args.modify_output:
            odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
            if args.asr_model_conf:
                asr_model = asrtts(idim, odim, args)
            else:
                asr_model = asrtts(idim, odim, args)
            asr_model.load_state_dict(remove_output_layer(torch.load(args.asr_model),
                                                          odim, args.eprojs,
                                                          args.dunits, 'asr'), strict=False)
        else:
            asr_model.load_state_dict(torch.load(args.asr_model), strict=False)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        asr_model.rnnlm = rnnlm
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
    elif args.freeze == "att":
        asr_model, size = freeze_parameters(asr_model, 16, "enc", "dec")
        logging.info("no of parameters frozen are: " + str(size))

    else:
        logging.warn("Only Disney is frozen")

    # setup loss functions
    # need to specify a loss function (loss_fn) to compute the expected
    # loss
    if args.expected_loss == 'tts':
        from espnet.nets.pytorch_backend.e2e_asrtts import load_tacotron_loss
        # assert args.tts_model, "Need to provide --tts-model and set --expected-loss tts"
        loss_fn = load_tacotron_loss(args.tts_model_conf, args.tts_model, args, asr_model.reporter)
    elif args.expected_loss == 'none':
        loss_fn = None
    else:
        raise NotImplementedError(
            'Unknown expected loss: %s' % args.expected_loss
        )
    asr2tts_model = asrtts(idim, odim, args, predictor=asr_model,
                           loss_fn=loss_fn, rnnlm=rnnlm, asr2tts=True, reporter=asr_model.reporter)

    if loss_fn is not None:
        tts_model = loss_fn.model
        tts2asr_model = Tacotron2ASRLoss(loss_fn.model, asr_model, args,
                                         reporter=asr_model.reporter,
                                         weight=args.teacher_weight)
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
        asr2tts_model = torch.nn.DataParallel(asr2tts_model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    asr2tts_model = asr2tts_model.to(device)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            tts2asr_model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(tts2asr_model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))
    # Setup a converter
    converter = CustomConverter(asr_model.subsample[0], args.use_speaker_embedding)

    # read json data
    train_json = []
    for idx in range(len(args.train_json)):
        with open(args.train_json[idx], 'rb') as f:
            train_json.append(json.load(f)['utts'])
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train_subsets = []
    for tj in train_json:
        train_subsets.append(make_batchset(tj, args.batch_size, args.maxlen_in,
                                           args.maxlen_out, args.minibatches,
                                           min_batch_size=args.ngpu if args.ngpu > 1 else 1))
    train = merge_batchsets(train_subsets, shuffle=True)
    valid = make_batchset(valid_json, args.batch_size, args.maxlen_in, args.maxlen_out,
                          args.minibatches, min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    load_tr = LoadInputsAndTargetsASRTTS(
        mode='asr', use_speaker_embedding=args.use_speaker_embedding,
        load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    # hack to make batchsze argument as 1
    # actual batchsize is included in a list
    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, load_cv),
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
        ASRConverter(asr_model.subsample[0]),
        CustomASRTTSConverter(args.use_speaker_embedding),
        device,
        args.ngpu,
        alpha=args.alpha,
        zero_att=args.zero_att)

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
            att_vis_fn = asr_model.module.calculate_all_attentions
        else:
            att_vis_fn = asr_model.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=device),
            trigger=(1, 'epoch'))

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
    if mtl_mode != 'ctc':
        trainer.extend(extensions.snapshot_object(asr2tts_model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))
        trainer.extend(extensions.snapshot_object(asr_model, 'model.acc.asr.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode != 'ctc':
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
    model = E2E(idim, odim, train_args)
    torch_load(args.model, model)
    model.recog_args = args
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
                nbest_hyps = model.recognize(feat, args, train_args.char_list, rnnlm)
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
                nbest_hyps = model.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)
                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
