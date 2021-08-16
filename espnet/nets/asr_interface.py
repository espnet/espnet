"""ASR Interface module."""
import argparse

from espnet.bin.asr_train import get_parser
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.fill_missing_args import fill_missing_args


class ASRInterface:
    """ASR Interface for ESPnet model implementation."""

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

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize x for evaluation.

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        raise NotImplementedError("recognize method is not implemented")

    def recognize_batch(self, x, recog_args, char_list=None, rnnlm=None):
        """Beam search implementation for batch.

        :param torch.Tensor x: encoder hidden state sequences (B, Tmax, Henc)
        :param namespace recog_args: argument namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        raise NotImplementedError("Batch decoding is not supported yet.")

    def calculate_all_attentions(self, xs, ilens, ys):
        """Calculate attention.

        :param list xs: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        """
        raise NotImplementedError("calculate_all_attentions method is not implemented")

    def calculate_all_ctc_probs(self, xs, ilens, ys):
        """Calculate CTC probability.

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: CTC probabilities (B, Tmax, vocab)
        :rtype: float ndarray
        """
        raise NotImplementedError("calculate_all_ctc_probs method is not implemented")

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        from espnet.asr.asr_utils import PlotAttentionReport

        return PlotAttentionReport

    @property
    def ctc_plot_class(self):
        """Get CTC plot class."""
        from espnet.asr.asr_utils import PlotCTCReport

        return PlotCTCReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        raise NotImplementedError(
            "get_total_subsampling_factor method is not implemented"
        )

    def encode(self, feat):
        """Encode feature in `beam_search` (optional).

        Args:
            x (numpy.ndarray): input feature (T, D)
        Returns:
            torch.Tensor for pytorch, chainer.Variable for chainer:
                encoded feature (T, D)

        """
        raise NotImplementedError("encode method is not implemented")

    def scorers(self):
        """Get scorers for `beam_search` (optional).

        Returns:
            dict[str, ScorerInterface]: dict of `ScorerInterface` objects

        """
        raise NotImplementedError("decoders method is not implemented")


predefined_asr = {
    "pytorch": {
        "rnn": "espnet.nets.pytorch_backend.e2e_asr:E2E",
        "transducer": "espnet.nets.pytorch_backend.e2e_asr_transducer:E2E",
        "transformer": "espnet.nets.pytorch_backend.e2e_asr_transformer:E2E",
        "conformer": "espnet.nets.pytorch_backend.e2e_asr_conformer:E2E",
    },
    "chainer": {
        "rnn": "espnet.nets.chainer_backend.e2e_asr:E2E",
        "transformer": "espnet.nets.chainer_backend.e2e_asr_transformer:E2E",
    },
}


def dynamic_import_asr(module, backend):
    """Import ASR models dynamically.

    Args:
        module (str): module_name:class_name or alias in `predefined_asr`
        backend (str): NN backend. e.g., pytorch, chainer

    Returns:
        type: ASR class

    """
    model_class = dynamic_import(module, predefined_asr.get(backend, dict()))
    assert issubclass(
        model_class, ASRInterface
    ), f"{module} does not implement ASRInterface"
    return model_class
