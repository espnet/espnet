#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import matplotlib.pyplot as plt
import numpy

from espnet.asr import asr_utils


def _plot_and_save_attention(att_w, filename, xtokens=None, ytokens=None):
    # dynamically import matplotlib due to not found error
    from matplotlib.ticker import MaxNLocator
    import os

    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        # plt.subplot(1, len(att_w), h)
        ax.imshow(aw.astype(numpy.float32), aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Labels for major ticks
        if xtokens is not None:
            ax.set_xticks(numpy.linspace(0, len(xtokens) - 1, len(xtokens)))
            ax.set_xticks(numpy.linspace(0, len(xtokens) - 1, 1), minor=True)
            ax.set_xticklabels(xtokens + [""], rotation=40)
        if ytokens is not None:
            ax.set_yticks(numpy.linspace(0, len(ytokens) - 1, len(ytokens)))
            ax.set_yticks(numpy.linspace(0, len(ytokens) - 1, 1), minor=True)
            ax.set_yticklabels(ytokens + [""])
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    plot.savefig(filename)
    plt.clf()


def plot_multi_head_attention(
    data,
    attn_dict,
    outdir,
    suffix="png",
    savefn=savefig,
    ikey="input",
    iaxis=0,
    okey="output",
    oaxis=0,
):
    """Plot multi head attentions.

    :param dict data: utts info from json file
    :param dict[str, torch.Tensor] attn_dict: multi head attention dict.
        values should be torch.Tensor (head, input_length, output_length)
    :param str outdir: dir to save fig
    :param str suffix: filename suffix including image type (e.g., png)
    :param savefn: function to save

    """
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.%s.%s" % (outdir, data[idx][0], name, suffix)
            dec_len = int(data[idx][1][okey][oaxis]["shape"][0])
            enc_len = int(data[idx][1][ikey][iaxis]["shape"][0])
            xtokens, ytokens = None, None
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
                # for MT
                if "token" in data[idx][1][ikey][iaxis].keys():
                    xtokens = data[idx][1][ikey][iaxis]["token"].split()
                    ytokens = xtokens[:]
            elif "decoder" in name:
                if "self" in name:
                    att_w = att_w[:, : dec_len + 1, : dec_len + 1]  # +1 for <sos>
                else:
                    att_w = att_w[:, : dec_len + 1, :enc_len]  # +1 for <sos>
                    # for MT
                    if "token" in data[idx][1][ikey][iaxis].keys():
                        xtokens = data[idx][1][ikey][iaxis]["token"].split()
                # for ASR/ST/MT
                if "token" in data[idx][1][okey][oaxis].keys():
                    ytokens = ["<sos>"] + data[idx][1][okey][oaxis]["token"].split()
                    if "self" in name:
                        xtokens = ytokens[:]
            else:
                logging.warning("unknown name for shaping attention")
            fig = _plot_and_save_attention(att_w, filename, xtokens, ytokens)
            savefn(fig, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def plotfn(self, *args, **kwargs):
        kwargs["ikey"] = self.ikey
        kwargs["iaxis"] = self.iaxis
        kwargs["okey"] = self.okey
        kwargs["oaxis"] = self.oaxis
        plot_multi_head_attention(*args, **kwargs)

    def __call__(self, trainer):
        attn_dict = self.get_attention_weights()
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        self.plotfn(self.data, attn_dict, self.outdir, suffix, savefig)

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        elif isinstance(batch, dict):
            att_ws = self.att_vis_fn(**batch)
        return att_ws

    def log_attentions(self, logger, step):
        def log_fig(plot, filename):
            from os.path import basename

            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        attn_dict = self.get_attention_weights()
        self.plotfn(self.data, attn_dict, self.outdir, "", log_fig)
