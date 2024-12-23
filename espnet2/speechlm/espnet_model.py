#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


@typechecked
class ESPnetSpeechLMModel(AbsESPnetModel):
    """
        ESPnetSpeechLMModel is a language model for speech processing that leverages a core
    language model (CoreLM) to generate and evaluate sequences. It is a subclass of
    the AbsESPnetModel and provides a forward method to process input sequences.

    Attributes:
        corelm (AbsCoreLM): An instance of a core language model that performs the
            actual sequence processing.
        extract_feats_in_collect_stats (bool): A flag indicating whether to extract
            features while collecting statistics.

    Args:
        corelm (AbsCoreLM): An instance of the core language model to be used.
        extract_feats_in_collect_stats (bool, optional): Whether to extract features
            in the collect statistics phase. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple containing
            the loss, statistics, and weight computed during the forward pass.

    Raises:
        NotImplementedError: If the collect_feats method is called, as it is not
            implemented in this class.

    Examples:
        >>> model = ESPnetSpeechLMModel(corelm)
        >>> dec_seq = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> dec_seq_lengths = torch.tensor([3, 3])
        >>> loss, stats, weight = model.forward(dec_seq, dec_seq_lengths)

    Note:
        This model is designed for use in the ESPnet framework for speech language
        modeling tasks.

    Todo:
        Implement the collect_feats method to allow feature extraction during the
        statistics collection phase.
    """

    def __init__(
        self,
        corelm: AbsCoreLM,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()

        self.corelm = corelm
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
                Performs the forward pass of the ESPnetSpeechLMModel, computing the loss,
        statistics, and weights based on the provided input sequences.

        Args:
            dec_seq (torch.Tensor): The decoded sequence tensor of shape (B, T_dec).
            dec_seq_lengths (torch.Tensor): A tensor containing the lengths of the
                decoded sequences of shape (B,).
            **kwargs: Additional keyword arguments, which may include:
                enc_seq (torch.Tensor, optional): The encoded sequence tensor of shape
                    (B, T_enc).
                enc_seq_lengths (torch.Tensor, optional): A tensor containing the lengths
                    of the encoded sequences of shape (B,).
                prefix_len (torch.Tensor, optional): A tensor representing the length of
                    the prefix for the sequence.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - loss (torch.Tensor): The computed loss value.
                - stats (Dict[str, torch.Tensor]): A dictionary containing various
                    statistics.
                - weight (torch.Tensor): The computed weight values.

        Raises:
            NotImplementedError: If called inappropriately or without required
                components.

        Examples:
            >>> model = ESPnetSpeechLMModel(corelm)
            >>> dec_seq = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> dec_seq_lengths = torch.tensor([3, 3])
            >>> loss, stats, weight = model.forward(dec_seq, dec_seq_lengths)

        Note:
            Ensure that the `corelm` is properly initialized and configured before
            invoking this method.
        """

        enc_seq = kwargs.get("enc_seq", None)
        enc_seq_lengths = kwargs.get("enc_seq_lengths", None)
        prefix_len = kwargs.get("prefix_len", None)
        if prefix_len is not None:
            prefix_len = prefix_len.squeeze(1)

        loss, stats, weight = self.corelm(
            dec_seq,
            dec_seq_lengths,
            enc_seq,
            enc_seq_lengths,
            prefix_len,
        )

        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight

    def collect_feats(self, **kwargs):
        """
                Collects features from the model. This method is currently not implemented.

        Args:
            **kwargs: Additional keyword arguments that may be required for future
                implementation.

        Raises:
            NotImplementedError: This method is a placeholder and should be
                implemented in subclasses.

        Examples:
            To use this method, you would typically call it on an instance of
            `ESPnetSpeechLMModel`, like so:

            ```python
            model = ESPnetSpeechLMModel(corelm)
            features = model.collect_feats()
            ```

        Note:
            This method is expected to be overridden in derived classes to provide
            the actual feature collection functionality.
        """
        raise NotImplementedError
