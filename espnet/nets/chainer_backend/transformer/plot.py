# encoding: utf-8
"""Class Declaration of Transformer's Attention Plot."""

from espnet.asr import asr_utils
import logging
import matplotlib.pyplot as plt


def savefig(plot, filename):
    """Save a figure."""
    plot.savefig(filename)
    plt.clf()


def _plot_and_save_attention(att_w, filename):
    """Plot and save an attention."""
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


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png", savefn=savefig):
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
    """Plot an attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_attentions):
        Function of attention visualization.
        data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter): Function to convert data.
        device (int | torch.device): Device.
        reverse (bool): If True, input and output length are reversed.
        ikey (str): Key to access input (for ASR ikey="input", for MT ikey="output".)
        iaxis (int): Dimension to access input (for ASR iaxis=0, for MT iaxis=1.)
        okey (str): Key to access output (for ASR okey="input", MT okay="output".)

    """

    def __call__(self, trainer):
        """Plot and save an image file of att_ws matrix."""
        attn_dict = self.get_attention_weights()
        suffix = "ep.{.updater.epoch}.png".format(trainer)
        plot_multi_head_attention(
            self.data, attn_dict, self.outdir, suffix, savefig)

    def get_attention_weights(self):
        """Return attention weights.

        Returns:
            numpy.ndarray: attention weights.float. Its shape would be
                differ from backend.
                * pytorch-> 1) multi-head case => (B, H, Lmax, Tmax), 2) other case => (B, Lmax, Tmax).
                * chainer-> (B, Lmax, Tmax)

        """
        batch = self.converter([self.transform(self.data)], self.device)
        return self.att_vis_fn(*batch)

    def log_attentions(self, logger, step):
        """Add image files of att_ws matrix to the tensorboard."""
        def log_fig(plot, filename):
            from os.path import basename
            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        attn_dict = self.get_attention_weights()
        plot_multi_head_attention(
            self.data, attn_dict, self.outdir, "", log_fig)
