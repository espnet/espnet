"""Default Recurrent Neural Network Languge Model in `lm_train.py`."""

from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.e2e_asr import to_device
from espnet.nets.scorer_interface import BatchScorerInterface


class DefaultRNNLM(BatchScorerInterface, LMInterface, nn.Module):
    """Default RNNLM for `LMInterface` Implementation.

    Note:
        PyTorch seems to have memory leak when one GPU compute this after data parallel.
        If parallel GPUs compute this, it seems to be fine.
        See also https://github.com/espnet/espnet/issues/1075

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments to command line argument parser."""
        parser.add_argument(
            "--type",
            type=str,
            default="lstm",
            nargs="?",
            choices=["lstm", "gru"],
            help="Which type of RNN to use",
        )
        parser.add_argument(
            "--layer", "-l", type=int, default=2, help="Number of hidden layers"
        )
        parser.add_argument(
            "--unit", "-u", type=int, default=650, help="Number of hidden units"
        )
        parser.add_argument(
            "--embed-unit",
            default=None,
            help="Number of hidden units in embedding layer, "
            "if it is not specified, it keeps the same number with hidden units.",
        )
        parser.add_argument(
            "--dropout-rate", type=float, default=0.5, help="dropout probability"
        )
        return parser

    def __init__(self, n_vocab, args):
        """Initialize class.

        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`

        """
        nn.Module.__init__(self)
        # NOTE: for a compatibility with less than 0.5.0 version models
        dropout_rate = getattr(args, "dropout_rate", 0.0)
        # NOTE: for a compatibility with less than 0.6.1 version models
        embed_unit = getattr(args, "embed_unit", None)
        self.model = ClassifierWithState(
            RNNLM(n_vocab, args.layer, args.unit, embed_unit, args.type, dropout_rate)
        )

    def state_dict(self):
        """Dump state dict."""
        return self.model.state_dict()

    def load_state_dict(self, d):
        """Load state dict."""
        self.model.load_state_dict(d)

    def forward(self, x, t):
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)
            t (torch.Tensor): Target ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used
            in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        loss = 0
        logp = 0
        count = torch.tensor(0).long()
        state = None
        batch_size, sequence_length = x.shape
        for i in range(sequence_length):
            # Compute the loss at this time step and accumulate it
            state, loss_batch = self.model(state, x[:, i], t[:, i])
            non_zeros = torch.sum(x[:, i] != 0, dtype=loss_batch.dtype)
            loss += loss_batch.mean() * non_zeros
            logp += torch.sum(loss_batch * non_zeros)
            count += int(non_zeros)
        return loss / batch_size, loss, count.to(loss.device)

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
        new_state, scores = self.model.predict(state, y[-1].unsqueeze(0))
        return scores.squeeze(0), new_state

    def final_score(self, state):
        """Score eos.

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return self.model.final(state)

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = self.model.predictor.n_layers
        if self.model.predictor.typ == "lstm":
            keys = ("c", "h")
        else:
            keys = ("h",)

        if states[0] is None:
            states = None
        else:
            # transpose state of [batch, key, layer] into [key, layer, batch]
            states = {
                k: [
                    torch.stack([states[b][k][i] for b in range(n_batch)])
                    for i in range(n_layers)
                ]
                for k in keys
            }
        states, logp = self.model.predict(states, ys[:, -1])

        # transpose state of [key, layer, batch] into [batch, key, layer]
        return (
            logp,
            [
                {k: [states[k][i][b] for i in range(n_layers)] for k in keys}
                for b in range(n_batch)
            ],
        )


class ClassifierWithState(nn.Module):
    """A wrapper for pytorch RNNLM."""

    def __init__(
        self, predictor, lossfun=nn.CrossEntropyLoss(reduction="none"), label_key=-1
    ):
        """Initialize class.

        :param torch.nn.Module predictor : The RNNLM
        :param function lossfun : The loss function to use
        :param int/str label_key :

        """
        if not (isinstance(label_key, (int, str))):
            raise TypeError("label_key must be int or str, but is %s" % type(label_key))
        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.label_key = label_key
        self.predictor = predictor

    def forward(self, state, *args, **kwargs):
        """Compute the loss value for an input and label pair.

        Notes:
            It also computes accuracy and stores it to the attribute.
            When ``label_key`` is ``int``, the corresponding element in ``args``
            is treated as ground truth labels. And when it is ``str``, the
            element in ``kwargs`` is used.
            The all elements of ``args`` and ``kwargs`` except the groundtruth
            labels are features.
            It feeds features to the predictor and compare the result
            with ground truth labels.

        :param torch.Tensor state : the LM state
        :param list[torch.Tensor] args : Input minibatch
        :param dict[torch.Tensor] kwargs : Input minibatch
        :return loss value
        :rtype torch.Tensor

        """
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = "Label key %d is out of bounds" % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[: self.label_key] + args[self.label_key + 1 :]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        state, self.y = self.predictor(state, *args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        return state, self.loss

    def predict(self, state, x):
        """Predict log probabilities for given state and input x using the predictor.

        :param torch.Tensor state : The current state
        :param torch.Tensor x : The input
        :return a tuple (new state, log prob vector)
        :rtype (torch.Tensor, torch.Tensor)
        """
        if hasattr(self.predictor, "normalized") and self.predictor.normalized:
            return self.predictor(state, x)
        else:
            state, z = self.predictor(state, x)
            return state, F.log_softmax(z, dim=1)

    def buff_predict(self, state, x, n):
        """Predict new tokens from buffered inputs."""
        if self.predictor.__class__.__name__ == "RNNLM":
            return self.predict(state, x)

        new_state = []
        new_log_y = []
        for i in range(n):
            state_i = None if state is None else state[i]
            state_i, log_y = self.predict(state_i, x[i].unsqueeze(0))
            new_state.append(state_i)
            new_log_y.append(log_y)

        return new_state, torch.cat(new_log_y)

    def final(self, state, index=None):
        """Predict final log probabilities for given state using the predictor.

        :param state: The state
        :return The final log probabilities
        :rtype torch.Tensor
        """
        if hasattr(self.predictor, "final"):
            if index is not None:
                return self.predictor.final(state[index])
            else:
                return self.predictor.final(state)
        else:
            return 0.0


# Definition of a recurrent net for language modeling
class RNNLM(nn.Module):
    """A pytorch RNNLM."""

    def __init__(
        self, n_vocab, n_layers, n_units, n_embed=None, typ="lstm", dropout_rate=0.5
    ):
        """Initialize class.

        :param int n_vocab: The size of the vocabulary
        :param int n_layers: The number of layers to create
        :param int n_units: The number of units per layer
        :param str typ: The RNN type
        """
        super(RNNLM, self).__init__()
        if n_embed is None:
            n_embed = n_units
        self.embed = nn.Embedding(n_vocab, n_embed)
        if typ == "lstm":
            self.rnn = nn.ModuleList(
                [nn.LSTMCell(n_embed, n_units)]
                + [nn.LSTMCell(n_units, n_units) for _ in range(n_layers - 1)]
            )
        else:
            self.rnn = nn.ModuleList(
                [nn.GRUCell(n_embed, n_units)]
                + [nn.GRUCell(n_units, n_units) for _ in range(n_layers - 1)]
            )

        self.dropout = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_layers + 1)]
        )
        self.lo = nn.Linear(n_units, n_vocab)
        self.n_layers = n_layers
        self.n_units = n_units
        self.typ = typ

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def zero_state(self, batchsize):
        """Initialize state."""
        p = next(self.parameters())
        return torch.zeros(batchsize, self.n_units).to(device=p.device, dtype=p.dtype)

    def forward(self, state, x):
        """Forward neural networks."""
        if state is None:
            h = [
                to_device(self, self.zero_state(x.size(0)))
                for n in range(self.n_layers)
            ]
            state = {"h": h}
            if self.typ == "lstm":
                c = [
                    to_device(self, self.zero_state(x.size(0)))
                    for n in range(self.n_layers)
                ]
                state = {"c": c, "h": h}

        h = [None] * self.n_layers
        emb = self.embed(x)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            h[0], c[0] = self.rnn[0](
                self.dropout[0](emb), (state["h"][0], state["c"][0])
            )
            for n in range(1, self.n_layers):
                h[n], c[n] = self.rnn[n](
                    self.dropout[n](h[n - 1]), (state["h"][n], state["c"][n])
                )
            state = {"c": c, "h": h}
        else:
            h[0] = self.rnn[0](self.dropout[0](emb), state["h"][0])
            for n in range(1, self.n_layers):
                h[n] = self.rnn[n](self.dropout[n](h[n - 1]), state["h"][n])
            state = {"h": h}
        y = self.lo(self.dropout[-1](h[-1]))
        return state, y
