"""Sequential implementation of Recurrent Neural Network Language Model."""
from typing import Tuple

import torch
import torch.nn as nn

from espnet2.lm.lm_interface import LMInterface


class SequentialRNNLM(LMInterface, torch.nn.Module):
    """Sequential RNNLM.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    """
    def __init__(self,
                 ntoken: int = 100,
                 unit: int = 650,
                 nlayers: int = 2,
                 dropout_rate: float = 0.5,
                 tie_weights: bool = False,
                 rnn_type: str = 'lstm',
                 ):
        """Initialize class.

        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`

        """
        torch.nn.Module.__init__(self)

        ninp = unit
        nhid = unit
        rnn_type = rnn_type.upper()

        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            rnn_class = getattr(nn, rnn_type)
            self.rnn = rnn_class(ninp, nhid, nlayers, dropout=dropout_rate)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              dropout=dropout_rate)
        self.decoder = nn.Linear(nhid, ntoken)

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
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),
                                           output.size(2)))
        return decoded.view(output.size(0), output.size(1),
                            decoded.size(1)), hidden

    def init_state(self, x):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        bsz = 1
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y, new_state = self._before_loss(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state
