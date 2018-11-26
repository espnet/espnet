#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import torch

# chainer related
import chainer

from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer
from chainer import training
from chainer.training import extension

# io related
import kaldi_io_py

# matplotlib related
import matplotlib
matplotlib.use('Agg')


# * -------------------- training iterator related -------------------- *
def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, min_batch_size=1):
    """Make batch set from json dictionary

    :param dict data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: mininum batch size (for multi-gpu)
    :return: list of batches
    """
    # sort it by input lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['input'][0]['shape'][0]), reverse=True)
    logging.info('# utts: ' + str(len(sorted_data)))

    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError("#utts is less than min_batch_size.")

    # make list of minibatches
    minibatches = []

    start = 0
    while True:
        ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
        olen = int(sorted_data[start][1]['output'][0]['shape'][0])
        factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        # change batchsize depending on the input and output length
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(min_batches, .) avoids batchsize = 0
        bs = max(min_batch_size, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + bs)
        minibatch = sorted_data[start:end]

        # check each batch is more than minimum batchsize
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i] for i in np.random.randint(0, start, mod)]
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # for debugging
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatches)))

    return minibatches

def get_output(output_name, utter):
    """ Returns the output corresponding to a given output name, for when there
    are multiple outputs. Example output_names include "grapheme" and
    "phn" (phoneme).

    Better would be to have json[utter_name]["output"] be a dictionary, but for
    now this function is to remove hardcoding of magic numbers like 0 for
    graphemes."""

    for output in utter["output"]:
        if output["name"] == output_name:
            return output["tokenid"].split()

    # If we're here, something has likely gone wrong
    logging.warn("output_name ({}) not found for utter ({})".format(
            output_name, utter))

# TODO This is very corpus-specific. Really this sort of think should go in the
# mkjson.py recipe so that the language is a field in data.json.
def uttid2lang(uttid):
    """ Returns the language given an utterance ID. Utterance IDs look
    something like this: "105_94168_A_20120127_071423_043760-turkish". Given
    this input, the output would be "turkish"
    """
    #return uttid.split("-")[-1]

    """ Changing this so we deal with lang-codes at the start of the uttid. In
    the above example it would be 105."""
    #return uttid.split("_")[0]

    """ For the CMU Wilderness dataset, its more like
        B27___22_Revelation__SWESFVN2DA_00028 """

    #lang = uttid.split("_")[-2][:6]
    lang = uttid[-16:-10]

    logging.info("Extracting langcode from uttid: {} -> {}".format(uttid, lang))

    return lang

def load_inputs_and_targets(batch, phoneme_objective_weight, lang2id):
    """Function to load inputs and targets from list of dicts

    :param list batch: list of dict which is subset of loaded data.json
    :return: list of input feature sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of target grapheme token id sequences [(L_1), (L_2), ..., (L_B)]
    :rtype: list of int ndarray
    :return: list of target phoneme token id sequences [(L_1), (L_2), ..., (L_B)]
    :rtype: list of int ndarray
    """
    # load acoustic features and target sequence of token ids
    xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]
    grapheme_ys = [get_output("grapheme", b[1]) for b in batch]

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(grapheme_ys[i]) > 0, range(len(xs)))
    # sort in input lengths
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
            len(xs), len(nonzero_sorted_idx)))

    # remove zero-length samples
    xs = [xs[i] for i in nonzero_sorted_idx]
    grapheme_ys = [np.fromiter(map(int, grapheme_ys[i]), dtype=np.int64) for i in nonzero_sorted_idx]

    # Gather phoneme targets
    if phoneme_objective_weight > 0.0:
        phoneme_ys = [get_output("phn", b[1]) for b in batch]
        phoneme_ys = [np.fromiter(map(int, phoneme_ys[i]), dtype=np.int64) for i in nonzero_sorted_idx]
    else:
        phoneme_ys = None


    # Gather language targets
    uttids = [b[0] for b in batch]
    uttids = [uttids[i] for i in nonzero_sorted_idx]
    logging.info("uttids: {}".format(uttids))
    try:
        lang_ys = np.fromiter([lang2id[uttid2lang(uttid)] for uttid in uttids],
                              dtype=np.int64)
    except TypeError:
        # Then the language of the utterance isn't in our list of languages, or
        # we don't even have a list of languages. In which case language
        # prediction shouldn't be part for training.
        lang_ys = None
    except KeyError:
        # Then the language of the utterance isn't in our list of languages, or
        # we don't even have a list of languages. In which case language
        # prediction shouldn't be part for training.
        lang_ys = None
    logging.info("lang_ys: {}".format(lang_ys))

    return xs, grapheme_ys, phoneme_ys, lang_ys

# * -------------------- chainer extension related -------------------- *
class CompareValueTrigger(object):
    '''Trigger invoked when key value getting bigger or lower than before

    Args:
        key (str): Key of value.
        compare_fn: Function to compare the values.
        trigger: Trigger that decide the comparison interval

    '''

    def __init__(self, key, compare_fn, trigger=(1, 'epoch')):
        self._key = key
        self._best_value = None
        self._interval_trigger = training.util.get_trigger(trigger)
        self._init_summary()
        self._compare_fn = compare_fn

    def __call__(self, trainer):
        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._best_value is None:
            # initialize best value
            self._best_value = value
            return False
        elif self._compare_fn(self._best_value, value):
            return True
        else:
            self._best_value = value
            return False

    def _init_summary(self):
        self._summary = chainer.reporter.DictSummary()


class PlotAttentionReport(extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param function converter: function to convert data
    :param int device: device id
    :param bool reverse: If True, input and output length are reversed
    """

    def __init__(self, att_vis_fn, data, outdir, converter, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad, lang_ys = self.converter([self.converter.transform(self.data)], self.device)
        att_ws = self.att_vis_fn(xs_pad, ilens, grapheme_ys_pad, phoneme_ys_pad)
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            if self.reverse:
                dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
                enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
            else:
                dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
                enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
            if len(att_w.shape) == 3:
                att_w = att_w[:, :dec_len, :enc_len]
            else:
                att_w = att_w[:dec_len, :enc_len]
            self._plot_and_save_attention(att_w, filename.format(trainer))

    def _plot_and_save_attention(self, att_w, filename):
        # dynamically import matplotlib due to not found error
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    '''Extension to restore snapshot'''
    @training.make_extension(trigger=(1, 'epoch'))
    def restore_snapshot(trainer):
        _restore_snapshot(model, snapshot, load_fn)

    return restore_snapshot


def _restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    load_fn(snapshot, model)
    logging.info('restored from ' + str(snapshot))


def adadelta_eps_decay(eps_decay):
    '''Extension to perform adadelta eps decay'''
    @training.make_extension(trigger=(1, 'epoch'))
    def adadelta_eps_decay(trainer):
        _adadelta_eps_decay(trainer, eps_decay)

    return adadelta_eps_decay


def _adadelta_eps_decay(trainer, eps_decay):
    optimizer = trainer.updater.get_optimizer('main')
    # for chainer
    if hasattr(optimizer, 'eps'):
        current_eps = optimizer.eps
        setattr(optimizer, 'eps', current_eps * eps_decay)
        logging.info('adadelta eps decayed to ' + str(optimizer.eps))
    # pytorch
    else:
        for p in optimizer.param_groups:
            p["eps"] *= eps_decay
            logging.info('adadelta eps decayed to ' + str(p["eps"]))


def torch_snapshot(savefun=torch.save,
                   filename='snapshot.ep.{.updater.epoch}'):
    """Returns a trainer extension to take snapshots of the trainer for pytorch."""
    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def torch_snapshot(trainer):
        _torch_snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return torch_snapshot


def _torch_snapshot_object(trainer, target, filename, savefun):
    # make snapshot_dict dictionary
    s = DictionarySerializer()
    s.save(trainer)
    if hasattr(trainer.updater.model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.model.model, "module"):
            model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.model, "module"):
            model_state_dict = trainer.updater.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "model": model_state_dict,
        "optimizer": trainer.updater.get_optimizer('main').state_dict()
    }

    # save snapshot dictionary
    fn = filename.format(trainer)
    prefix = 'tmp' + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefun(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)


# * -------------------- general -------------------- *
class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def __getitem__(self, name):
        return self.obj[name]

    def __len__(self):
        return len(self.obj)

    def fields(self):
        return self.obj

    def items(self):
        return self.obj.items()

    def keys(self):
        return self.obj.keys()


def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json)

    :param str model_path: model path
    :param str conf_path: optional model config path
    """

    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        return json.load(f, object_hook=AttributeDict)


def chainer_load(path, model):
    """Function to load chainer model parameters

    :param str path: model file or snapshot file to be loaded
    :param chainer.Chain model: chainer model
    """
    if 'snapshot' in path:
        chainer.serializers.load_npz(path, model, path='updater/model:main/')
    else:
        chainer.serializers.load_npz(path, model)


def torch_save(path, model):
    """Function to save torch model states

    :param str path: file path to be saved
    :param torch.nn.Module model: torch model
    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    """Function to load torch model states

    :param str path: model file or snapshot file to be loaded
    :param torch.nn.Module model: torch model
    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


def torch_resume(snapshot_path, trainer, restore_trainer):
    """Function to resume from snapshot for pytorch

    :param str snapshot_path: snapshot file path
    :param instance trainer: chainer trainer instance
    :param bool restore_trainer: Resume training with the same chainer Trainer
    that was used for initial model training. Set to false if adapting with a
    different trainer.
    """
    # load snapshot
    logging.info("snapshot_path: {}".format(snapshot_path))
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)
    logging.info("snapshot_dict: {}".format(snapshot_dict))

    logging.info(restore_trainer)
    if restore_trainer:
        # restore trainer states
        d = NpzDeserializer(snapshot_dict['trainer'])
        d.load(trainer)

    # restore model states
    if hasattr(trainer.updater.model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.model.model, "module"):
            trainer.updater.model.model.module.load_state_dict(snapshot_dict['model'])
        else:
            trainer.updater.model.model.load_state_dict(snapshot_dict['model'])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.model, "module"):
            trainer.updater.model.module.load_state_dict(snapshot_dict['model'])
        else:
            trainer.updater.model.load_state_dict(snapshot_dict['model'])
            #trainer.updater.model.load_state_dict(snapshot_dict)

    # retore optimizer states
    trainer.updater.get_optimizer('main').load_state_dict(snapshot_dict['optimizer'])

    # delete opened snapshot
    del snapshot_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list, output_type):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        for output in js['output']:
            if output['name'] == output_type:
                # copy ground-truth
                out_dic = dict(output.items())

                # update name
                out_dic['name'] += '[%d]' % n

                # add recognition results
                out_dic['rec_text'] = rec_text
                out_dic['rec_token'] = rec_token
                out_dic['rec_tokenid'] = rec_tokenid
                out_dic['score'] = score

                # add to list of N-best result dicts
                new_js['output'].append(out_dic)

                # show 1-best result
                if n == 1:
                    logging.info('groundtruth: %s' % out_dic['text'])
                    logging.info('prediction : %s' % out_dic['rec_text'])

    return new_js
