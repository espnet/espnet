import chainer


class Reporter(chainer.Chain):
    def report(self, dicts):
        for d in dicts:
            chainer.reporter.report(d, self)


class TTSInterface(object):
    """TTS Interface for ESPnet model implementation"""

    @staticmethod
    def add_arguments(parser):
        return parser

    def __init__(self):
        self.reporter = Reporter()

    def forward(self, *args, **kwargs):
        """Calculate TTS forward propagation

        :return: loss value
        :rtype: torch.Tensor
        """
        raise NotImplementedError("forward method is not implemented")

    def inference(self, *args, **kwargs):
        """Generates the sequence of features given the sequences of characters

        :return: the sequence of features (L, odim)
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        :return: the sequence of attention weight (L, T)
        :rtype: torch.Tensor
        """
        raise NotImplementedError("inference method is not implemented")

    def calculate_all_attentions(self, *args, **kwargs):
        """Calculate TTS attention weights

        :return: attention weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    @property
    def attention_plot_class(self):
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport

    @property
    def base_plot_keys(self):
        """base key names to plot during training. keys should match what `chainer.reporter` reports

        if you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        :rtype list[str] plot_keys: base keys to plot during training
        """
        return ['loss']
