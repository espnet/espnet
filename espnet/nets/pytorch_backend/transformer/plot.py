"""pytorch_backend/transformer/plot module."""

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os

import numpy

from espnet.asr import asr_utils


def _plot_and_save_attention(att_w, filename, xtokens=None, ytokens=None):
    """Plot and save attention."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

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
            ax.set_xticks(numpy.linspace(0, len(xtokens), len(xtokens) + 1))
            ax.set_xticks(numpy.linspace(0, len(xtokens), 1), minor=True)
            ax.set_xticklabels(xtokens + [""], rotation=40)
        if ytokens is not None:
            ax.set_yticks(numpy.linspace(0, len(ytokens), len(ytokens) + 1))
            ax.set_yticks(numpy.linspace(0, len(ytokens), 1), minor=True)
            ax.set_yticklabels(ytokens + [""])
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    """Save figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot.savefig(filename)
    plt.clf()


def plot_multi_head_attention(
    data,
    uttid_list,
    attn_dict,
    outdir,
    suffix="png",
    savefn=savefig,
    ikey="input",
    iaxis=0,
    okey="output",
    oaxis=0,
    subsampling_factor=4,
):
    """Plot multi head attentions.

    :param dict data: utts info from json file
    :param List uttid_list: utterance IDs
    :param dict[str, torch.Tensor] attn_dict: multi head attention dict.
        values should be torch.Tensor (head, input_length, output_length)
    :param str outdir: dir to save fig
    :param str suffix: filename suffix including image type (e.g., png)
    :param savefn: function to save
    :param str ikey: key to access input
    :param int iaxis: dimension to access input
    :param str okey: key to access output
    :param int oaxis: dimension to access output
    :param subsampling_factor: subsampling factor in encoder

    """
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            data_i = data[uttid_list[idx]]
            filename = "%s/%s.%s.%s" % (outdir, uttid_list[idx], name, suffix)
            dec_len = int(data_i[okey][oaxis]["shape"][0]) + 1  # +1 for <eos>
            enc_len = int(data_i[ikey][iaxis]["shape"][0])
            is_mt = "token" in data_i[ikey][iaxis].keys()
            # for ASR/ST
            if not is_mt:
                enc_len //= subsampling_factor
            xtokens, ytokens = None, None
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
                # for MT
                if is_mt:
                    xtokens = data_i[ikey][iaxis]["token"].split()
                    ytokens = xtokens[:]
            elif "decoder" in name:
                if "self" in name:
                    # self-attention
                    att_w = att_w[:, :dec_len, :dec_len]
                    if "token" in data_i[okey][oaxis].keys():
                        ytokens = data_i[okey][oaxis]["token"].split() + ["<eos>"]
                        xtokens = ["<sos>"] + data_i[okey][oaxis]["token"].split()
                else:
                    # cross-attention
                    att_w = att_w[:, :dec_len, :enc_len]
                    if "token" in data_i[okey][oaxis].keys():
                        ytokens = data_i[okey][oaxis]["token"].split() + ["<eos>"]
                    # for MT
                    if is_mt:
                        xtokens = data_i[ikey][iaxis]["token"].split()
            else:
                logging.warning("unknown name for shaping attention")
            fig = _plot_and_save_attention(att_w, filename, xtokens, ytokens)
            savefn(fig, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    """Plot Attention Report class."""

    def plotfn(self, *args, **kwargs):
        """Process plot function."""
        kwargs["ikey"] = self.ikey
        kwargs["iaxis"] = self.iaxis
        kwargs["okey"] = self.okey
        kwargs["oaxis"] = self.oaxis
        kwargs["subsampling_factor"] = self.factor
        plot_multi_head_attention(*args, **kwargs)

    def __call__(self, trainer):
        """Process call function."""
        attn_dict, uttid_list = self.get_attention_weights()
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        self.plotfn(self.data_dict, uttid_list, attn_dict, self.outdir, suffix, savefig)

    def get_attention_weights(self):
        """Get attention weights."""
        return_batch, uttid_list = self.transform(self.data, return_uttid=True)
        batch = self.converter([return_batch], self.device)
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        elif isinstance(batch, dict):
            att_ws = self.att_vis_fn(**batch)
        return att_ws, uttid_list

    def log_attentions(self, logger, step):
        """Log attentions."""

        def log_fig(plot, filename):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            logger.add_figure(os.path.basename(filename), plot, step)
            plt.clf()

        attn_dict, uttid_list = self.get_attention_weights()
        self.plotfn(self.data_dict, uttid_list, attn_dict, self.outdir, "", log_fig)
