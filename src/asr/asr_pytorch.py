#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import pickle
import sys

# chainer related
import chainer

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch

# spnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import converter_kaldi
from asr_utils import delete_feat
from asr_utils import load_labeldict
from asr_utils import make_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from e2e_asr_attctc_th import E2E
from e2e_asr_attctc_th import Loss

# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_pytorch
import lm_pytorch

# matplotlib related
import matplotlib
matplotlib.use('Agg')


class PytorchSeqEvaluaterKaldi(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(PytorchSeqEvaluaterKaldi, self).__init__(
            iterator, target, converter=converter, device=device)
        self.model = model

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

        for batch in it:
            observation = {}
            if torch.__version__ != "0.3.1":
                torch.set_grad_enabled(False)
            with reporter_module.report_scope(observation):
                # read scp files
                # x: original json with loaded features
                #    will be converted to chainer variable later
                x = self.converter(batch)
                self.model.eval()
                self.model(x)
                delete_feat(x)

            if torch.__version__ != "0.3.1":
                torch.set_grad_enabled(True)

            summary.add(observation)

        self.model.train()

        return summary.compute_mean()


class PytorchSeqUpdaterKaldi(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device):
        super(PytorchSeqUpdaterKaldi, self).__init__(
            train_iter, optimizer, converter=converter, device=None)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.num_gpu = len(device)

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
        if len(batch[0]) < self.num_gpu:
            logging.warning('batch size is less than number of gpus. Ignored')
            return
        x = self.converter(batch)

        # Compute the loss at this time step and accumulate it
        loss = 1. / self.num_gpu * self.model(x)
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.num_gpu > 1:
            loss.backward(torch.ones(self.num_gpu))  # Backprop
        else:
            loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        delete_feat(x)


class DataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids, dim):
        r"""Scatter with support for kwargs dictionary"""
        if len(inputs) == 1:
            inputs = inputs[0]
        avg = int(math.ceil(len(inputs) * 1. / len(device_ids)))
        # inputs = scatter(inputs, device_ids, dim) if inputs else []
        inputs = [[inputs[i:i + avg]] for i in range(0, len(inputs), avg)]
        kwargs = torch.nn.scatter(kwargs, device_ids, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.dim)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

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

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    # TODO(nelson) remove in future
    if 'input' not in valid_json[utts[0]]:
        logging.error(
            "input file format (json) is modified, please redo"
            "stage 2: Dictionary and Json Data Preparation")
        sys.exit(1)
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
    model = Loss(e2e, args.mtlalpha, phoneme_objective_weight=args.phoneme_objective_weight)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((idim, grapheme_odim, phoneme_odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    reporter = model.reporter
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model = DataParallel(model, device_ids=gpu_id)
        model.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    else:
        gpu_id = [-1]

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

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
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = PytorchSeqUpdaterKaldi(
        model, args.grad_clip, train_iter, optimizer, converter=converter_kaldi, device=gpu_id)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        if ngpu > 1:
            model.module.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        else:
            model.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(PytorchSeqEvaluaterKaldi(
        model, valid_iter, reporter, converter=converter_kaldi, device=gpu_id))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        data = converter_kaldi([data], device=gpu_id)
        trainer.extend(PlotAttentionReport(model, data, args.outdir + "/att_ws"), trigger=(1, 'epoch'))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att',
                                          'main/loss_phn', 'validation/main/loss_phn'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    def torch_save(path, _):
        if ngpu > 1:
            torch.save(model.module.state_dict(), path)
            torch.save(model.module, path + ".pkl")
        else:
            torch.save(model.state_dict(), path)
            torch.save(model, path + ".pkl")

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # epsilon decay in the optimizer
    def torch_load(path, obj):
        if ngpu > 1:
            model.module.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path))
        return obj
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
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(100, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(100, 'iteration'))

    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, grapheme_odim, phoneme_odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Phoneme objective weight set to zero.
    #train_args.phoneme_objective_weight = 0.0

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e = E2E(idim, grapheme_odim, train_args, phoneme_odim=phoneme_odim)
    model = Loss(e2e, train_args.mtlalpha)
    # We're going to need to specify the phoneme objective weight, aren't we? or are we?
    #model = Loss(e2e, args.mtlalpha, phoneme_objective_weight=args.phoneme_objective_weight)

    def cpu_loader(storage, location):
        return storage

    def remove_dataparallel(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        return new_state_dict

    #logging.info("args.model: {}".format(args))
    a = torch.load(args.model, map_location=cpu_loader)
    b = remove_dataparallel(a)
    logging.info("remove_dataparallel:{}".format(b))
    model.load_state_dict(b)
    #model.load_state_dict(remove_dataparallel(torch.load(args.model, map_location=cpu_loader)))

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), 650))
        rnnlm.load_state_dict(torch.load(args.rnnlm, map_location=cpu_loader))
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        if not args.word_dict:
            logging.error('word dictionary file is not specified for the word RNNLM.')
            sys.exit(1)

        word_dict = load_labeldict(args.word_dict)
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(len(word_dict), 650))
        word_rnnlm.load_state_dict(torch.load(args.word_rnnlm, map_location=cpu_loader))
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
        recog_json = json.load(f)['utts']

    if train_args.phoneme_objective_weight > 0:
        assert args.phoneme_dict
        with open(args.phoneme_dict) as f:
            # The zero is because of the CTC blank symbol and because the
            # phoneme inventory list starts indexing from 1.
            phn_inv_list = [0]+[line.split()[0] for line in f.readlines()]

    new_json = {}
    for name in recog_json.keys():
        feat = kaldi_io_py.read_mat(recog_json[name]['input'][0]['feat'])
        nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm)
        # get 1best and remove sos
        y_hat = nbest_hyps[0]['yseq'][1:]
        for output in recog_json[name]['output']:
            if output['name'] == "grapheme":
                y_true = map(int, output['tokenid'].split())
                break

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

        # Adding recognition results to the JSON. Start by copying our original
        # JSON outputs.
        logging.debug("dump token id")
        grapheme_output = get_output("grapheme", name, recog_json)
        grapheme_out_dict = copy.deepcopy(grapheme_output)

        # TODO(karita) make consistent to chainer as idx[0] not idx
        grapheme_out_dict['rec_tokenid'] = " ".join([str(idx) for idx in y_hat])
        logging.debug("dump token")
        grapheme_out_dict['rec_token'] = " ".join(seq_hat)
        logging.debug("dump text")
        grapheme_out_dict['rec_text'] = seq_hat_text

        new_json[name]['output'] = [grapheme_out_dict]

        if train_args.phoneme_objective_weight > 0:
            phn_output = get_output("phn", name, recog_json)
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

            new_json[name]['output'].append(phn_out_dict)

        # TODO(nelson): Modify this part when saving more than 1 hyp is enabled
        # add n-best recognition results with scores
        if args.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] = " ".join([str(idx) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = hyp['score']

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))
