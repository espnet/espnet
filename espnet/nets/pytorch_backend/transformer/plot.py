import logging

import matplotlib.pyplot as plt

from espnet.asr import asr_utils


def _plot_and_save_attention(att_w, filename):
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
        ax.imshow(aw, aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    plot.savefig(filename)
    plt.clf()


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png", savefn=savefig):
    """Plot multi head attentions

    :param dict data: utts info from json file
    :param dict[str, torch.Tensor] attn_dict: multi head attention dict.
        values should be torch.Tensor (head, input_length, output_length)
    :param str outdir: dir to save fig
    :param str suffix: filename suffix including image type (e.g., png)
    :param savefn: function to save
    """
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
            fig = _plot_and_save_attention(att_w, filename)
            savefn(fig, filename)


class PlotAttentionReport(asr_utils.PlotAttentionReport):
    def __call__(self, trainer):
        attn_dict = self.get_attention_weights()
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        plot_multi_head_attention(
            self.data, attn_dict, self.outdir, suffix, savefig)

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        return self.att_vis_fn(*batch)

    def log_attentions(self, logger, step):
        def log_fig(plot, filename):
            from os.path import basename
            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        attn_dict = self.get_attention_weights()
        plot_multi_head_attention(
            self.data, attn_dict, self.outdir, "", log_fig)
