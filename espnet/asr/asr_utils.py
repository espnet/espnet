#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
# matplotlib related
import os
import shutil
import tempfile

# chainer related
import chainer

from chainer import training
from chainer.training import extension

from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer

# io related
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')


# * -------------------- training iterator related -------------------- *


class CompareValueTrigger(object):
    """Trigger invoked when key value getting bigger or lower than before.

    Args:
        key (str) : Key of value.
        compare_fn ((float, float) -> bool) : Function to compare the values.
        trigger (tuple(int, str)) : Trigger that decide the comparison interval.

    """

    def __init__(self, key, compare_fn, trigger=(1, 'epoch')):
        self._key = key
        self._best_value = None
        self._interval_trigger = training.util.get_trigger(trigger)
        self._init_summary()
        self._compare_fn = compare_fn

    def __call__(self, trainer):
        """Get value related to the key and compare with current value."""
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
    """Plot attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_attentions): Function of attention visualization.
        data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter): Function to convert data.
        device (int | torch.device): Device.
        reverse (bool): If True, input and output length are reversed.

    """

    def __init__(self, att_vis_fn, data, outdir, converter, transform, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        """Plot and save image file of att_ws matrix."""
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            att_w = self.get_attention_weight(idx, att_w)
            self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        """Add image files of att_ws matrix to the tensorboard."""
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            att_w = self.get_attention_weight(idx, att_w)
            plot = self.draw_attention_plot(att_w)
            logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
            plot.clf()

    def get_attention_weights(self):
        """Return attention weights.

        Returns:
            numpy.ndarray: attention weights.float. Its shape would be
                differ from backend.
                * pytorch-> 1) multi-head case => attention weights (B, H, Lmax, Tmax),
                            2) other case => attention weights (B, Lmax, Tmax).
                * chainer-> (B, Lmax, Tmax)

        """
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        else:
            att_ws = self.att_vis_fn(**batch)
        return att_ws

    def get_attention_weight(self, idx, att_w):
        """Transform attention matrix with regard to self.reverse."""
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
        return att_w

    def draw_attention_plot(self, att_w):
        """Plot the att_w matrix.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
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
        return plt

    def _plot_and_save_attention(self, att_w, filename):
        plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


def restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    """Extension to restore snapshot.

    Returns:
        An extension function.

    """
    @training.make_extension(trigger=(1, 'epoch'))
    def restore_snapshot(trainer):
        _restore_snapshot(model, snapshot, load_fn)

    return restore_snapshot


def _restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    load_fn(snapshot, model)
    logging.info('restored from ' + str(snapshot))


def adadelta_eps_decay(eps_decay):
    """Extension to perform adadelta eps decay.

    Returns:
        An extension function.

    """
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
    """Extension to take snapshot of the trainer for pytorch.

    Returns:
        An extension function.

    """
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


def add_gradient_noise(model, epoch, eta):
    """Adds noise from a std normal distribution to the gradients.

    Args:
        model (Torch model): model.
        iteration (int): number of iteration.
        eta (float): {0.01,0.3,1.0}

    """
    scale_factor = 0.55
    sigma = eta / epoch**scale_factor
    for param in model.parameters():
        if param.grad is not None:
            _shape = param.grad.size()
            noise = sigma * torch.randn(_shape).cuda()
            param.grad += noise


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
    """Get model config information by reading a model config file (model.json).

    Args:
        model_path (str): model path
        conf_path (str): optional model config path

    Returns:
        list[int, int, dict[str, Any]]: config information loaded from json file.

    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        return json.load(f, object_hook=AttributeDict)


def chainer_load(path, model):
    """Load chainer model parameters.

    Args:
        path (str): model file or snapshot file to be loaded
        model (chainer.Chain): chainer model

    """
    if 'snapshot' in path:
        chainer.serializers.load_npz(path, model, path='updater/model:main/')
    else:
        chainer.serializers.load_npz(path, model)


def torch_save(path, model):
    """Save torch model states.

    Args:
        path (str): file path to be saved
        model (torch.nn.Module): torch model

    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def snapshot_object(target, filename):
    """Returns a trainer extension to take snapshots of a given object.

    Args:
        target (model): Object to serialize.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth: `str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.

    Returns:
        An extension function.

    """
    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def snapshot_object(trainer):
        torch_save(os.path.join(trainer.out, filename.format(trainer)), target)

    return snapshot_object


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): model file or snapshot file to be loaded.
        model (torch.nn.Module): torch model.

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


def torch_resume(snapshot_path, trainer):
    """Resume from snapshot for pytorch.

    Args:
        snapshot_path (str): snapshot file path.
        trainer (chainer.training.Trainer): Chainer's trainer instance

    """
    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

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

    # retore optimizer states
    trainer.updater.get_optimizer('main').load_state_dict(snapshot_dict['optimizer'])

    # delete opened snapshot
    del snapshot_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): recognition hypothesis.
        char_list (list[str]): list of characters.

    Returns:
        tuple(str, str, str, float)

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


def add_results_to_json(js, nbest_hyps, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]): List of hypothesis for multi_speakers: nutts x nspkrs.
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        if len(js['output']) > 0:
            out_dic = dict(js['output'][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {'name': ''}

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
            if 'text' in out_dic.keys():
                logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])

    return new_js


def plot_spectrogram(plt, spec, mode='db', fs=None, frame_shift=None,
                     bottom=True, left=True, right=True, top=False,
                     labelbottom=True, labelleft=True, labelright=True,
                     labeltop=False, cmap='inferno'):
    """Plot spectrogram using matplotlib.

    Args:
        plt (matplotlib.pyplot): pyplot object.
        spec (numpy.ndarray): Input stft (Freq, Time)
        mode (str): db or linear.
        fs (int): Sample frequency. To convert y-axis to kHz unit.
        frame_shift (int): The frame shift of stft. To convert x-axis to second unit.
        bottom (bool):Whether to draw the respective ticks.
        left (bool):
        right (bool):
        top (bool):
        labelbottom (bool):Whether to draw the respective tick labels.
        labelleft (bool):
        labelright (bool):
        labeltop (bool):
        cmap (str): colormap defined in matplotlib

    """
    spec = np.abs(spec)
    if mode == 'db':
        x = 20 * np.log10(spec + np.finfo(spec.dtype).eps)
    elif mode == 'linear':
        x = spec
    else:
        raise ValueError(mode)

    if fs is not None:
        ytop = fs / 2000
        ylabel = 'kHz'
    else:
        ytop = x.shape[0]
        ylabel = 'bin'

    if frame_shift is not None and fs is not None:
        xtop = x.shape[1] * frame_shift / fs
        xlabel = 's'
    else:
        xtop = x.shape[1]
        xlabel = 'frame'

    extent = (0, xtop, 0, ytop)
    plt.imshow(x[::-1], cmap=cmap, extent=extent)

    if labelbottom:
        plt.xlabel('time [{}]'.format(xlabel))
    if labelleft:
        plt.ylabel('freq [{}]'.format(ylabel))
    plt.colorbar().set_label('{}'.format(mode))

    plt.tick_params(bottom=bottom, left=left, right=right, top=top,
                    labelbottom=labelbottom, labelleft=labelleft,
                    labelright=labelright, labeltop=labeltop)
    plt.axis('auto')
