"""Sequential implementation of Recurrent Neural Network Language Model."""
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.lm.abs_model import AbsLM


class SequentialRNNLM(AbsLM):
    """Sequential RNNLM.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    """

    def __init__(
        self,
        vocab_size: int,
        unit: int = 650,
        nhid: int = None,
        nlayers: int = 2,
        dropout_rate: float = 0.0,
        tie_weights: bool = False,
        rnn_type: str = "lstm",
        ignore_id: int = 0,
    ):
        assert check_argument_types()
        super().__init__()

        ninp = unit
        if nhid is None:
            nhid = unit
        rnn_type = rnn_type.upper()

        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=ignore_id)
        if rnn_type in ["LSTM", "GRU"]:
            rnn_class = getattr(nn, rnn_type)
            self.rnn = rnn_class(
                ninp, nhid, nlayers, dropout=dropout_rate, batch_first=True
            )
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp,
                nhid,
                nlayers,
                nonlinearity=nonlinearity,
                dropout=dropout_rate,
                batch_first=True,
            )
        self.decoder = nn.Linear(nhid, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016) https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers:
        # A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            hidden,
        )

    def init_state(self, x):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        bsz = 1
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def score(
        self,
        y: torch.Tensor,
        state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Score new token.

        Args:
            y: 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x: 2D encoder feature that generates ys.

        Returns:
            Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y, new_state = self(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state
