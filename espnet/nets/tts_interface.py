# -*- coding: utf-8 -*-

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TTS Interface realted modules."""

from espnet.asr.asr_utils import torch_load

try:
    import chainer
except ImportError:
    Reporter = None
else:

    class Reporter(chainer.Chain):
        """Reporter module."""

        def report(self, dicts):
            """Report values from a given dict."""
            for d in dicts:
                chainer.reporter.report(d, self)


class TTSInterface(object):
    """TTS Interface for ESPnet model implementation."""

    @staticmethod
    def add_arguments(parser):
        """Add model specific argments to parser."""
        return parser

    def __init__(self):
        """Initilize TTS module."""
        self.reporter = Reporter()

    def forward(self, *args, **kwargs):
        """Calculate TTS forward propagation.

        Returns:
            Tensor: Loss value.

        """
        raise NotImplementedError("forward method is not implemented")

    def inference(self, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Returns:
            Tensor: The sequence of generated features (L, odim).
            Tensor: The sequence of stop probabilities (L,).
            Tensor: The sequence of attention weights (L, T).

        """
        raise NotImplementedError("inference method is not implemented")

    def calculate_all_attentions(self, *args, **kwargs):
        """Calculate TTS attention weights.

        Args:
            Tensor: Batch of attention weights (B, Lmax, Tmax).

        """
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    def load_pretrained_model(self, model_path):
        """Load pretrained model parameters."""
        torch_load(model_path, self)

    @property
    def attention_plot_class(self):
        """Plot attention weights."""
        from espnet.asr.asr_utils import PlotAttentionReport

        return PlotAttentionReport

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training.

        The keys should match what `chainer.reporter` reports.
        if you add the key `loss`,
        the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss`
        and `validation/main/loss` values.

        Returns:
            list[str]:  Base keys to plot during training.

        """
        return ["loss"]
