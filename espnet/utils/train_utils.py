import copy
import json
import logging
import os

import chainer
from chainer.training import extension

from tensorboardX import SummaryWriter

from espnet.utils.tensorboard_logger import TensorboardLogger

REPORT_INTERVAL = 100


def check_early_stop(trainer, epochs):
    end_epoch = trainer.updater.get_iterator('main').epoch
    if end_epoch < (epochs - 1):
        logging.warning("Hit early stop at epoch " + str(
            end_epoch) + "\nYou can change the patience or set it to 0 to run all epochs")


def add_tensorboard(trainer, tensorboard_dir, attention_reporter=None):
    if tensorboard_dir is not None and tensorboard_dir != "":
        writer = SummaryWriter(log_dir=tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, attention_reporter))


def add_early_stop(trainer, args):
    if args.patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(monitor=args.early_stop_criterion,
                                                                              patients=args.patience,
                                                                              max_trigger=(args.epochs, 'epoch'))


def load_jsons(args):
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    return train_json, valid_json


def add_attention_report(trainer, model, args, valid_json, converter, device, reversePAR=False):
    if args.num_save_attention > 0 and (not hasattr(args, 'mtlalpha') or args.mtlalpha != 1.0):
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
        else:
            att_vis_fn = model.calculate_all_attentions
        att_reporter = PlotAttentionReport(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, device=device, reverse=reversePAR)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None
    return att_reporter


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
