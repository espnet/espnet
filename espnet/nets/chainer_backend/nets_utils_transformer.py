# encoding: utf-8

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.asr import asr_utils

import logging
import numpy as np


class PositionalEncoding(chainer.Chain):
    def __init__(self, n_units, dropout=0.1, length=5000):
        # Implementation described in the paper
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        posi_block = np.arange(
            0, length, dtype=np.float32)[:, None]
        unit_block = np.exp(
            np.arange(0, n_units, 2, dtype=np.float32) * -(np.log(10000.) / n_units))
        self.pe = np.zeros((length, n_units), dtype=np.float32)
        self.pe[:, ::2] = np.sin(posi_block * unit_block)
        self.pe[:, 1::2] = np.cos(posi_block * unit_block)
        self.scale = np.sqrt(n_units)

    def __call__(self, e):
        length = e.shape[1]
        e = e * self.scale + self.xp.array(self.pe[:length])
        return F.dropout(e, self.dropout)


class LayerNorm(L.LayerNormalization):
    def __init__(self, dims, eps=1e-12):
        super(LayerNorm, self).__init__(size=dims, eps=eps)

    def __call__(self, e):
        return super(LayerNorm, self).__call__(e)


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units, d_units=0, dropout=0.1, initialW=None, initial_bias=None):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = d_units if d_units > 0 else n_units * 4
        with self.init_scope():
            stvd = 1. / np.sqrt(n_units)
            self.w_1 = L.Linear(n_units, n_inner_units,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            stvd = 1. / np.sqrt(n_inner_units)
            self.w_2 = L.Linear(n_inner_units, n_units,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.act = F.relu
        self.dropout = dropout

    def __call__(self, e):
        e = F.dropout(self.act(self.w_1(e)), self.dropout)
        return self.w_2(e)


def _plot_and_save_attention(att_w, filename):
    # dynamically import matplotlib due to not found error
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import os

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        ax.imshow(aw, aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(filename)


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png"):
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.%s.%s" % (
                outdir, data[idx][0], name, suffix)
            dec_len = int(data[idx][1]['output'][0]['shape'][0])
            enc_len = int(data[idx][1]['input'][0]['shape'][0])
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
            elif "decoder" in name:
                if "self" in name:
                    att_w = att_w[:, :dec_len, :dec_len]
                else:
                    att_w = att_w[:, :dec_len, :enc_len]
            else:
                logging.warning("unknown name for shaping attention")
            _plot_and_save_attention(att_w, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def __call__(self, trainer):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        attn_dict = self.att_vis_fn(*batch)
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        plot_multi_head_attention(self.data, attn_dict, self.outdir, suffix)
