# encoding: utf-8

from espnet.asr import asr_utils
import logging
import matplotlib.pyplot as plt


def savefig(plot, filename):
    """Save figure to given path.

    Args:
        filename (str): Output path for the image.

    """
    plot.savefig(filename)
    plt.clf()


class PositionalEncoding(chainer.Chain):
    """Positional encoding implementation.

    Args:
        n_units (int): Dimension of the model.
        dropout (float): Dropout rate.
        length (int): Seqense length of inputs.

    """

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
        """Compute positional encoding.

        Args:
            e (chainer.Variale): Input array.

        Returns
            chainer.Variable: Positional-encoded array.

        """
        length = e.shape[1]
        e = e * self.scale + self.xp.array(self.pe[:length])
        return F.dropout(e, self.dropout)


class LayerNorm(L.LayerNormalization):
    """Layer normalization.

    Args:
        dims (int): Size of input units.
        eps (float): Epsilon value for numerical stability of normalization.

    """

    def __init__(self, dims, eps=1e-12):
        super(LayerNorm, self).__init__(size=dims, eps=eps)

    def __call__(self, e):
        """Compute layer normalization.

        Args:
            e (chainer.Variable): Batch vectors. Shape of this value must be
                `(batch_size, unit_size)`.

        Returns:
            chainer.Variable: Output of the layer normalization.

        """
        return super(LayerNorm, self).__call__(e)


class FeedForwardLayer(chainer.Chain):
    """Feed Forward.

    Args:
        n_units (int): Dimension of the inputs/outputs of this layer.
        d_units (int): Dimension of the hidden layer.
        dropout (float): Dropout rate.
        initialW (Initializer): Initializer to initialize the weight.
        initial_bias (Initializer): Initializer to initialize the bias.

    """

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
        """Compute feed forward layer.

        Args:
            e (chainer.Variable): Input array.

        Returns:
            chainer.Variable: Output of the feed-forward network.

        """
        e = F.dropout(self.act(self.w_1(e)), self.dropout)
        return self.w_2(e)


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


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png", savefn=savefig):
    """Plot multi head attentions.

    Args:
        data (dict): Utts info from json file.
        attn_dict (Dict[str, chainer.Variable]): Multi head attention dict. (head, input_length, output_length)
        outdir (str): Directory to save figure.
        suffix (str): Filename suffix including image type. (e.g., png)
        savefn (function): Function to save.

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
