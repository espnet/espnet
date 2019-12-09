"""MT Interface module."""
import argparse

from espnet.bin.asr_train import get_parser
from espnet.utils.fill_missing_args import fill_missing_args


class MTInterface:
    """MT Interface for ESPnet model implementation."""

    @staticmethod
    def add_arguments(parser):
        """Add arguments to parser."""
        return parser

    @classmethod
    def build(cls, idim: int, odim: int, **kwargs):
        """Initialize this class with python-level args.

        Args:
            idim (int): The number of an input feature dim.
            odim (int): The number of output vocab.

        Returns:
            ASRinterface: A new instance of ASRInterface.

        """
        def wrap(parser):
            return get_parser(parser, required=False)

        args = argparse.Namespace(**kwargs)
        args = fill_missing_args(args, wrap)
        args = fill_missing_args(args, cls.add_arguments)
        return cls(idim, odim, args)

    def forward(self, xs, ilens, ys):
        """Compute loss for training.

        :param xs:
            For pytorch, batch of padded source sequences torch.Tensor (B, Tmax, idim)
            For chainer, list of source sequences chainer.Variable
        :param ilens: batch of lengths of source sequences (B)
            For pytorch, torch.Tensor
            For chainer, list of int
        :param ys:
            For pytorch, batch of padded source sequences torch.Tensor (B, Lmax)
            For chainer, list of source sequences chainer.Variable
        :return: loss value
        :rtype: torch.Tensor for pytorch, chainer.Variable for chainer
        """
        raise NotImplementedError("forward method is not implemented")

    def translate(self, x, trans_args, char_list=None, rnnlm=None):
        """Translate x for evaluation.

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace trans_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        raise NotImplementedError("translate method is not implemented")

    def translate_batch(self, x, trans_args, char_list=None, rnnlm=None):
        """Beam search implementation for batch.

        :param torch.Tensor x: encoder hidden state sequences (B, Tmax, Henc)
        :param namespace trans_args: argument namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        raise NotImplementedError("Batch decoding is not supported yet.")

    def calculate_all_attentions(self, xs, ilens, ys):
        """Caluculate attention.

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        """
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport
