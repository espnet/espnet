#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
from collections import defaultdict
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
from espnet.asr.asr_utils import uttid2lang
from espnet.nets.e2e_asr_th import E2E
from espnet.nets.e2e_asr_th import Loss
from espnet.nets.e2e_asr_th import pad_list

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
                    xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad, lang_ys = self.converter(batch, self.device)
                    self.model(xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()

def ganin_lambda(epoch, total_epochs):
    """ Sets the domain adaptation scaling factor on the basis of Ganin et al.
    2016. The learning rate progresses from 0 to 1 in a non-linear logarithmic
    manner."""
    progress = epoch / float(total_epochs)
    logging.info("Progress as decimal: {}".format(progress))
    lambda_ = 2/(1 + np.exp(-10*progress)) - 1
    logging.info("Ganin lambda: {}".format(lambda_))
    return lambda_

def shinohara_lambda(epoch, lambda_max=0.1):
    """ Sets the domain adaptation scaling factor on the basis of Shinohara
    2016. The learning rate progresses from 0 to lambda_max in a linear
    manner."""

    lambda_ = min(epoch/float(10), 1)*lambda_max
    logging.info("Shinohara lambda: {}".format(lambda_))
    return lambda_

class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, num_epochs,
                 predict_lang=None, predict_lang_alpha=None,
                 predict_lang_alpha_scheduler=None,
                 adapt=None):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.predict_lang = predict_lang
        self.predict_lang_alpha = predict_lang_alpha
        self.predict_lang_alpha_scheduler = predict_lang_alpha_scheduler
        self.adapt = adapt
        self.num_epochs = num_epochs

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad, lang_ys = self.converter(batch, self.device)
        # Compute the loss at this time step and accumulate it
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.ngpu > 1:
            loss = 1. / self.ngpu * self.model(xs_pad, ilens, grapheme_ys_pad,
                                               phoneme_ys_pad)
            loss.backward(loss.new_ones(self.ngpu))  # Backprop
        else:
            loss = self.model(xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad)
            loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()

        logging.info("predict_lang: {}".format(self.predict_lang))
        logging.info("predict_lang_alpha: {}".format(self.predict_lang_alpha))
        logging.info("epoch: {}".format(self.epoch))
        logging.info("epoch_detail: {}".format(self.epoch_detail))
        logging.info("is_new_epoch: {}".format(self.is_new_epoch))
        logging.info("previous_epoch_detail: {}".format(self.previous_epoch_detail))
        if self.adapt:
            # Then don't do language-based prediction and learning
            logging.info("Don't do language-based prediction and learning.")
            return
        if self.predict_lang: # Either normal prediction or adversarial
            # Now compute the lang loss (this redoes the encoding, which may be
            # slightly inefficient but should be fine for now).
            # (If performing adaptation, then we don't do adversarial language
            # prediction)
            optimizer.zero_grad()
            if self.ngpu > 1:
                lang_loss = 1. / self.ngpu * self.model.forward_langid(xs_pad, ilens, lang_ys)
                lang_loss.backward(torch.ones(self.ngpu))  # Backprop
            else:
                lang_loss = self.model.forward_langid(xs_pad, ilens, lang_ys)
                lang_loss.backward()  # Backprop
            lang_loss.detach()  # Truncate the graph
            logging.info("predict_lang: {}".format(self.predict_lang))


            if self.predict_lang == "adv":
                if self.predict_lang_alpha_scheduler == "ganin":
                    lambda_ = ganin_lambda(self.epoch, self.num_epochs)
                elif self.predict_lang_alpha_scheduler == "shinohara":
                    lambda_ = shinohara_lambda(self.epoch)
                elif self.predict_lang_alpha:
                    lambda_ = self.predict_lang_alpha
                    logging.info("Fixed lambda: {}".format(lambda_))
                else:
                    raise ValueError("""predict_lang_alpha_scheduler ({}) or
                        predict_lang_alpha ({}) not set to a valid
                        option.""".format(self.predict_lang_alpha_scheduler,
                                          self.predict_lang_alpha))
                if lambda_ != 0.0:
                    # Then it's adversarial and we should reverse gradients
                    for name, parameter in self.model.named_parameters():
                        logging.info("lambda_: {}".format(lambda_))
                        logging.info("name, parameter: {}".format(name, parameter))
                        logging.info("parameter.grad: {}".format(parameter.grad))
                        parameter.grad *= -1 * lambda_

                    # But reverse the lang_linear gradients again so that
                    # they're not adversarial (we just want the encoder to hide
                    # the language information, but we still want to try our
                    # best to predict the language)
                    self.model.lang_linear.bias.grad *= (-1 / lambda_)
                    self.model.lang_linear.weight.grad *= (-1 / lambda_)

            optimizer.step()


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, phoneme_objective_weight, langs, subsamping_factor=1):
        self.ignore_id = -1
        self.phoneme_objective_weight = phoneme_objective_weight
        self.subsamping_factor = subsamping_factor
        self.langs = langs
        if self.langs:
            self.lang2id = {lang: id_ for id_, lang in enumerate(self.langs)}
            self.id2lang = {id_: lang for id_, lang in enumerate(self.langs)}
        else:
            self.lang2id = None
            self.id2lang = None

    def transform(self, item):
        return load_inputs_and_targets(
                item, self.phoneme_objective_weight,
                self.lang2id)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, grapheme_ys, phoneme_ys, lang_ys = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        grapheme_ys_pad = pad_list([torch.from_numpy(y).long() for y in grapheme_ys], self.ignore_id).to(device)

        if self.phoneme_objective_weight > 0.0:
            assert phoneme_ys
            phoneme_ys_pad = pad_list([torch.from_numpy(y).long() for y in phoneme_ys], self.ignore_id).to(device)
        else:
            phoneme_ys_pad = None

        # lang_ys may be None if we don't want to train language prediction.
        if lang_ys is not None:
            lang_ys = torch.from_numpy(lang_ys).long().to(device)

        return xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad, lang_ys

class EspnetException(Exception):
    pass

class NoOdimException(EspnetException):
    pass

def get_odim(output_name, valid_json):
    """ Return the output dimension for a given output type.
    For example, output type might be 'phn' (phonemes) or 'grapheme'.

    Note this is based off the first utterance, so it's assumed the output
    dimension doesn't change across utterances in the JSON."""

    utts = list(valid_json.keys())
    for output in valid_json[utts[0]]['output']:
        if output['name'] == output_name:
            return int(output['shape'][1])
    # Raise an exception because we couldn't find the odim
    raise NoOdimException("Couldn't determine output dimension (odim) for output named '{}'".format(output_name))

def extract_langs(json):
    """ Determines the number of output languages."""

    # Create a list of languages observed by taking them from the utterance
    # name.
    utts = list(json.keys())
    langs = set()
    for utt in utts:
        langs.add(uttid2lang(utt))
    return langs

def get_output(output_name, utterance_name, json):
    """ Returns the dictionary corresponding to a given output_name in some
    espnet utterance JSON. For example. Example output_names include "grapheme" and
    "phn" (phoneme).

    Better would be to have json[utter_name]["output"] be a dictionary, but this
    function is to remove hardcoding of magic numbers like 0 for graphemes."""

    utts = list(json.keys())
    for output in json[utterance_name]["output"]:
        if output["name"] == output_name:
            return output

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

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    #langs = extract_langs(train_json)
    #langs = langs.union(set(range(20)))
    if args.langs_file:
        langs = set()
        with open(args.langs_file) as f:
            for line in f:
                langs.add(unicode(line.strip()))
        logging.error(langs)
    else:
        langs = None

    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    grapheme_odim = get_odim("grapheme", valid_json)
    logging.info('#input dims : ' + str(idim))
    logging.info('#grapheme output dims: ' + str(grapheme_odim))
    phoneme_odim = -1
    if args.phoneme_objective_weight > 0.0:
        phoneme_odim = get_odim("phn", valid_json)
        logging.info('#phoneme output dims: ' + str(phoneme_odim))

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
    if args.phoneme_objective_weight > 0.0:
        logging.info('Training with an additional phoneme transcription objective.')

    if args.pretrained_model:
        logging.info("Reading a pretrained model from " + args.pretrained_model)
        train_idim, train_odim, phoneme_odim, train_args = get_model_conf(args.pretrained_model)

        if train_args.phoneme_objective_weight > 0.0:
            e2e = E2E(train_idim, train_odim, train_args, phoneme_odim=phoneme_odim)
        else:
            e2e = E2E(train_idim, train_odim, train_args)

        if train_args.langs_file:
            langs = set()
            with open(args.langs_file) as f:
                for line in f:
                    langs.add(unicode(line.strip()))
            logging.error(langs)
        else:
            langs = None

        model = Loss(e2e, train_args.mtlalpha,
                phoneme_objective_weight=train_args.phoneme_objective_weight,
                langs=langs)

        model.load_state_dict(torch.load(args.pretrained_model,
                map_location=lambda storage, loc: storage))

    else:
        # specify model architecture
        if args.phoneme_objective_weight > 0.0:
            e2e = E2E(idim, grapheme_odim, args, phoneme_odim=phoneme_odim)
        else:
            e2e = E2E(idim, grapheme_odim, args)
        model = Loss(e2e, args.mtlalpha, 
                     phoneme_objective_weight=args.phoneme_objective_weight,
                     langs=langs)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch.load(args.rnnlm, rnnlm)
        e2e.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, grapheme_odim, phoneme_odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    #for parameter in model.parameters():
    #    logging.info("Model parameter: {}".format(parameter))
    logging.info("Model: {}".format(model))

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(args.phoneme_objective_weight, langs,
                                e2e.subsample[0])

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1)
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
        model, args.grad_clip, train_iter, optimizer, converter, device,
        args.ngpu, args.epochs, predict_lang=args.predict_lang,
        predict_lang_alpha=args.predict_lang_alpha,
        predict_lang_alpha_scheduler=args.predict_lang_alpha_scheduler,
        adapt=args.adapt)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer, args.restore_trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.predictor.calculate_all_attentions
        else:
            att_vis_fn = model.predictor.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, device=device), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att',
                                          'main/loss_phn', 'validation/main/loss_phn'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['main/acc_lang', 'validation/main/acc_lang'],
                                         'epoch', file_name='acc_lang.png'))

    # Save best models
    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
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
    try:
        idim, grapheme_odim, phoneme_odim, train_args = get_model_conf(args.model, args.model_conf)
    except ValueError:
        # Couldn't find phoneme_odim
        idim, grapheme_odim, train_args = get_model_conf(args.model, args.model_conf)
        logging.info("train_args.phoneme_objective_weight: {}".format(train_args.phoneme_objective_weight))
        if train_args.phoneme_objective_weight > 0.0:
            phoneme_odim = 118
        else:
            phoneme_odim = -1

    # read training json data just so we can extract the list of langs used in
    # language ID prediction
    if args.langs_file:
        langs = set()
        with open(args.langs_file) as f:
            for line in f:
                langs.add(line.strip())
    else:
        langs = None

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, grapheme_odim, train_args, phoneme_odim=phoneme_odim)
    model = Loss(e2e, train_args.mtlalpha, langs=langs)
    #model = Loss(e2e, train_args.mtlalpha,
    #             phoneme_objective_weight=args.phoneme_objective_weight)
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

    if args.encoder_states:
        logging.info("Storing encoder states.")
        logging.info("per-frame-ali: {}".format(args.per_frame_ali))
        per_frame_phns = dict()
        with open(args.per_frame_ali) as f:
            for line in f:
                sp = line.split()
                name = sp[0]
                phns = sp[1:]
                per_frame_phns[unicode(name)] = phns

        phn_units_path=("/export/b13/oadams/espnet-merge/egs/babel/"
                       "phoneme_objective/data/lang_1char/train_units.txt.phn")
        phn_units = []
        with open(phn_units_path, "r") as f:
            for line in f:
                phn_units.append(line.split()[0])

        NUM_ENCODER_STATES = None
        #target_phns = ["a", "i", "o", "6",]# "u", "@", "E",]
        target_phns = phn_units
        target_langs = ["102", "103", "104", "105", "106", "204", "206", "207"]
        encoder_states = defaultdict(list)
        uttids = defaultdict(list)
        for idx, name in enumerate(js.keys(), 1):
            lang = name.split("_")[0]
            uttids[lang].append(name)
        for lang in target_langs:
            write_enc_states(lang, target_phns,
                             js, uttids, per_frame_phns, e2e, train_args,
                             num_encoder_states=NUM_ENCODER_STATES)

        return

    # decode each utterance
    new_js = {}

    if args.batchsize is None:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
                nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list, "grapheme")
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
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list, "grapheme")

    if args.recog_phonemes:
        assert args.phoneme_dict
        with open(args.phoneme_dict) as f:
            # The zero is because of the CTC blank symbol and because the
            # phoneme inventory list starts indexing from 1.
            phn_inv_list = [0]+[line.split()[0] for line in f.readlines()]

            phn_output = get_output("phn", name, js)
            if phn_output:
                phn_out_dict = copy.deepcopy(phn_output)
                phn_true = phn_out_dict["token"]
                logging.info("ground truth phns: {}".format(phn_true))
            else:
                # Then there was no ground truth phonemes, so we create a new
                # output.
                phn_out_dict = {}
                phn_out_dict["name"] = "phn"

            # Then do basic one-best CTC phoneme decoding
            phn_hyps = e2e.recognize_phn(feat)
            phn_hat = [phn_inv_list[idx] for idx in phn_hyps]
            logging.info("predicted phns: {}".format(phn_hat))

            # Add phoneme-related info to the output JSON
            phn_out_dict['rec_tokenid'] = " ".join([str(idx) for idx in phn_hyps])
            phn_out_dict['rec_token'] = " ".join(phn_hat)

            new_js[name]['output'].append(phn_out_dict)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True, ensure_ascii=False).encode('utf_8'))


def write_enc_states(lang, tgt_phns,
                     js, uttids, per_frame_phns, e2e, train_args,
                     num_encoder_states=500):
    """ Writes the encoder states for a given language and phoneme. """

    logging.info("target_phns: {}".format(tgt_phns))
    encoder_states = defaultdict(list)
    for name in uttids[lang]:
        if name in per_frame_phns:
            logging.info("name: {}".format(name))
            feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
            logging.info("feat.shape: {}".format(feat.shape))
            logging.info("e2e.subsample: {}".format(e2e.subsample))
            h, lens = e2e.encode_from_feat(feat)
            logging.info("h.shape: {}".format(h.shape))
            logging.info("lens: {}".format(lens))
            logging.info("per_frame_phns: {}".format(
                         per_frame_phns[name]))
            logging.info("len(per_frame_phns): {}".format(
                         len(per_frame_phns[name])))

            i = 0
            phns = per_frame_phns[name]
            while i < len(phns):
                if phns[i] in tgt_phns:
                    tgt = phns[i]
                    start = i
                    while i < len(phns) and phns[i] == tgt:
                        i += 1
                    end = i - 1
                    # Midpoint between when the phoneme starts and ends.
                    mid = int((start + end)/2)
                    # Turn that midpoint into an index in the encoder
                    # states by scaling by the length of phonemes.
                    h_i = int((mid/float(len(phns)))*h.shape[1])
                    if num_encoder_states:
                        if len(encoder_states[tgt]) < num_encoder_states:
                            encoder_states[tgt].append(h[0,h_i,:].detach().numpy())
                    else:
                        encoder_states[tgt].append(h[0,h_i,:].detach().numpy())
                i += 1

            done = True
            for tgt in encoder_states:
                logging.info("len(encoder_states[{}]): {}".format(
                             tgt, len(encoder_states[tgt])))
                if num_encoder_states:
                    if len(encoder_states[tgt]) != num_encoder_states:
                        done = False
                # If num_encoder_states is set to None, then done=True, then we just keep
                # writing our matrix out for each utter.
            if done:
                for tgt in encoder_states:
                    #encoder_states[tgt] = np.array(encoder_states[tgt]).T
                    logging.info("writing encoder_states/{}_alpha{}beta{}_predict-lang-{}-{}_phn-{}_num{}_encoder_states.npy".format(lang,
                    train_args.mtlalpha, train_args.phoneme_objective_weight,
                    train_args.predict_lang, train_args.predict_lang_alpha_scheduler,
                    tgt, num_encoder_states))
                    np.save("dev_encoder_states/{}_alpha{}beta{}_predict-lang-{}-{}_phn-{}_num{}_encoder_states".format(lang,
                    train_args.mtlalpha, train_args.phoneme_objective_weight,
                    train_args.predict_lang, train_args.predict_lang_alpha_scheduler,
                    tgt, num_encoder_states), np.array(encoder_states[tgt]))
                if num_encoder_states:
                    return
