"""ST Interface module."""

from espnet.nets.asr_interface import ASRInterface


class STInterface(ASRInterface):
    """ST Interface for ESPnet model implementation."""

    def translate(self, x, trans_args, char_list=None, rnnlm=None):
        """Recognize x for evaluation.

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
