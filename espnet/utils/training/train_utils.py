import copy
import json
import logging
import os

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

from tensorboardX import SummaryWriter

from espnet.utils.training.tensorboard_logger import TensorboardLogger

from espnet.utils.pytorch_utils import torch_resume
from espnet.utils.pytorch_utils import torch_save

REPORT_INTERVAL = 100


def check_early_stop(trainer, epochs):
    """Checks if the training was stopped by an early stopping trigger and warns the user if it's the case

    :param trainer: The trainer used for training
    :param epochs: The maximum number of epochs
    """
    end_epoch = trainer.updater.get_iterator('main').epoch
    if end_epoch < (epochs - 1):
        logging.warning("Hit early stop at epoch " + str(
            end_epoch) + "\nYou can change the patience or set it to 0 to run all epochs")


def add_tensorboard(trainer, tensorboard_dir, attention_reporter=None):
    """Adds the tensorboard extension

    :param trainer: The trainer to add the extension to
    :param tensorboard_dir: The tensorboard log directory
    :param attention_reporter: The plot attention reporter
    """
    if tensorboard_dir is not None and tensorboard_dir != "":
        writer = SummaryWriter(log_dir=tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, attention_reporter))


def add_early_stop(trainer, args):
    """Adds an early stop trigger

    :param trainer: The trainer to add the trigger to
    :param args: The program arguments
    """
    if args.patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(monitor=args.early_stop_criterion,
                                                                              patients=args.patience,
                                                                              max_trigger=(args.epochs, 'epoch'))


def load_jsons(args):
    """Loads training and validation utterances from the json files

    :param args: the program arguments
    :return: training json, validation json
    """
    train_json = load_json(args.train_json)
    valid_json = load_json(args.valid_json)
    return train_json, valid_json


def load_json(json_file):
    """Loads the utterances data from a json file

    :param json_file: The json filepath
    :return: The json object
    """
    with open(json_file, 'rb') as f:
        return json.load(f)['utts']


def add_snapshot(trainer, model, savefun, mtl_mode=None):
    """Adds the model snapshot extension

    :param trainer: The trainer to add the extension to
    :param model: The model to save
    :param mtl_mode: The mtl mode
    :param savefun: The save function to use
    """
    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=savefun),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode is not None and mtl_mode is not 'ctc':
        trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=savefun),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))


def add_attention_report(trainer, model, args, valid_json, converter, device, reverse_par=False):
    """Adds the attention reporter extension

    :param trainer: The trainer to add the extension to
    :param model: The model to train
    :param args: The program arguments
    :param valid_json: The validation json
    :param converter: The batch converter
    :param device: The device to use
    :param reverse_par: If the input and output length should be reversed for the plot
    :return: the plot attention reporter
    """
    if args.num_save_attention > 0 and (not hasattr(args, 'mtlalpha') or args.mtlalpha != 1.0):
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
        else:
            att_vis_fn = model.calculate_all_attentions
        att_reporter = PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, device=device, reverse=reverse_par)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None
    return att_reporter


def add_plot_report(trainer, values):
    """Adds the plot report extension

    :param trainer: The trainer to add the extension to
    """
    for tup in values:
        trainer.extend(extensions.PlotReport(tup[1], 'epoch', file_name=tup[0] + '.png'))


def prepare_asr_tts_trainer(updater, evaluator, converter, model, valid_json, args, plot_keys, device, mtl_mode=None,
                            reverse_par=False):
    """Prepares a common trainer for asr and tts
    :param updater: The training updater
    :param evaluator: The training evaluator
    :param converter: The batch converter
    :param model: The model
    :param valid_json: The validation json
    :param args: The program arguments
    :param plot_keys: The keys to report for plotting
    :param device: The device to use
    :param mtl_mode: The mtl mode
    :param reverse_par: Whether to reverse the axis for the attention report
    :return: The trainer with common extensions
    """
    is_chainer = args.backend == 'chainer'
    savefun = chainer.serializers.save_npz if is_chainer else torch_save
    resume_fun = chainer.serializers.load_npz if is_chainer else torch_resume

    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if args.resume:
        logging.info('resumed from %s' % args.resume)
        resume_fun(args.resume, trainer)

    trainer.extend(evaluator)
    add_snapshot(trainer, model, savefun, mtl_mode)
    add_early_stop(trainer, args)
    add_plot_report(trainer, plot_keys)
    att_reporter = add_attention_report(trainer, model, args, valid_json, converter, device, reverse_par=reverse_par)
    add_tensorboard(trainer, args.tensorboard_dir, att_reporter)
    return trainer


class PlotAttentionReport(extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param CustomConverter converter: function to convert data
    :param int | torch.device device: device
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
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            att_w = self.get_attention_weight(idx, att_w)
            self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            att_w = self.get_attention_weight(idx, att_w)
            plot = self.draw_attention_plot(att_w)
            logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
            plot.clf()

    def get_attention_weights(self):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        att_ws = self.att_vis_fn(*batch)
        return att_ws

    def get_attention_weight(self, idx, att_w):
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


def write_conf(args, idim=None, odim=None):
    """Writes a model configuration

    :param args: the program arguments
    :param idim: the feature input dimension
    :param odim: the model output dimension
    """
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(
            json.dumps((idim, odim, vars(args)) if idim is not None else (vars(args)), indent=4, sort_keys=True).encode(
                'utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))
