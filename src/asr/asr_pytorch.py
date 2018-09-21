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
from asr_utils import adadelta_eps_decay
from asr_utils import add_results_to_json
from asr_utils import CompareValueTrigger
from asr_utils import get_model_conf
from asr_utils import load_inputs_and_targets
from asr_utils import make_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from asr_utils import torch_load
from asr_utils import torch_resume
from asr_utils import torch_save
from asr_utils import torch_snapshot
from asr_utils import uttid2lang
from e2e_asr_th import E2E
from e2e_asr_th import Loss
from e2e_asr_th import pad_list

# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_pytorch
import lm_pytorch

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

    # The core part of the update routine can be customized by overriding.
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

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu,
                 predict_lang=None, predict_lang_alpha=None):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.predict_lang = predict_lang
        self.predict_lang_alpha = predict_lang_alpha

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        xs_pad, ilens, grapheme_ys, phoneme_ys, lang_ys = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.ngpu > 1:
            loss = 1. / self.ngpu * self.model(*x)
            loss.backward(loss.new_ones(self.ngpu))  # Backprop
        else:
            loss = self.model(*x)
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
        if self.predict_lang: # Either normal prediction or adversarial
            # Now compute the lang loss (this redoes the encoding, which may be
            # slightly inefficient but should be fine for now).
            optimizer.zero_grad()
            if self.ngpu > 1:
                lang_loss = 1. / self.ngpu * self.model.forward_langid(xs_pad, ilens, lang_ys)
                lang_loss.backward(torch.ones(self.ngpu))  # Backprop
            else:
                lang_loss = self.model.forward_langid(xs_pad, ilens, lang_ys)
                lang_loss.backward()  # Backprop
            logging.info("predict_lang: {}".format(self.predict_lang))

            if self.predict_lang == "adv":
                assert self.predict_lang_alpha
                # Then it's adversarial and we should reverse gradients
                for name, parameter in self.model.named_parameters():
                #    logging.info("parameter {} grad: {}".format(
                #            name, parameter.grad))
                    parameter.grad *= -1 * self.predict_lang_alpha
                #    logging.info("parameter {} -grad: {}".format(
                #            name, parameter.grad))

#               # But reverse the lang_linear gradients again so that they're
                # not adversarial (we just want the encoder to hide the
                # language information, but we still want to try our best to predict the
                # language)
                self.model.lang_linear.bias.grad *= (-1 / self.predict_lang_alpha)
                self.model.lang_linear.weight.grad *= (-1 / self.predict_lang_alpha)
                #logging.info("lang_linear {}".format((self.model.lang_linear.bias.grad,
                #    self.model.lang_linear.weight.grad)))

            optimizer.step()


class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, phoneme_objective_weight, langs, subsamping_factor=1):
        self.ignore_id = -1
        self.phoneme_objective_weight = phoneme_objective_weight
        self.subsamping_factor = subsamping_factor
        self.langs = langs
        self.lang2id = {lang: id_ for id_, lang in enumerate(self.langs)}
        self.id2lang = {id_: lang for id_, lang in enumerate(self.langs)}

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

    langs = extract_langs(train_json)

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

    # specify model architecture
    if args.phoneme_objective_weight > 0.0:
        e2e = E2E(idim, grapheme_odim, args, phoneme_odim=phoneme_odim)
    else:
        e2e = E2E(idim, grapheme_odim, args)
    model = Loss(e2e, args.mtlalpha, 
                 phoneme_objective_weight=args.phoneme_objective_weight,
                 langs=langs)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, grapheme_odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
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
    converter = CustomConverter(args.phoneme_objective_weight, langs, e2e.subsample[0])

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.MultiprocessIterator(
        TransformDataset(train, converter.transform),
        batch_size=1, n_processes=1, n_prefetch=8, maxtasksperchild=20)
    valid_iter = chainer.iterators.SerialIterator(
        TransformDataset(valid, converter.transform),
        batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, converter, device, args.ngpu,
        predict_lang=args.predict_lang, predict_lang_alpha=args.predict_lang_alpha)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

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
    idim, grapheme_odim, phoneme_odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, grapheme_odim, train_args, phoneme_odim=phoneme_odim)
    model = Loss(e2e, train_args.mtlalpha,
                 phoneme_objective_weight=args.phoneme_objective_weight)
    torch_load(args.model, model)

    # read training json data just so we can extract the list of langs used in
    # language ID prediction
    if args.train_json:
        with open(args.train_json, 'rb') as f:
            train_json = json.load(f)['utts']
        langs = extract_langs(train_json)
    else:
        langs = None


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

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
            nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    if train_args.phoneme_objective_weight > 0:
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
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
