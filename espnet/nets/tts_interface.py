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

    def forward(self, *args, **kwargs):
        """Calculate TTS forward propagation"""
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


class TTSLossInterface(object):
    """TTS Loss Interface for ESPnet model implementation"""

    @staticmethod
    def add_arguments(parser):
        return parser

    def __init__(self):
        self.reporter = Reporter()

    def forward(self, *args, **kwargs):
        """TTS Loss forward computation

        :return: loss value
        :rtype: torch.Tensor
        """
        raise NotImplementedError("forward method is not implemented")
