"""Sequential implementation of Recurrent Neural Network Language Model."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM


class SequentialRNNLM(AbsLM):
    """
        Sequential implementation of Recurrent Neural Network Language Model.

    This class implements a Sequential RNN Language Model (RNNLM) using PyTorch.
    It supports different types of RNNs including LSTM, GRU, and standard RNNs with
    tanh or ReLU activations. The model can optionally tie the weights of the
    output layer with the embedding layer, which is a common technique to improve
    language models.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    Attributes:
        drop (nn.Dropout): Dropout layer for regularization.
        encoder (nn.Embedding): Embedding layer for input tokens.
        rnn (nn.Module): RNN layer (LSTM, GRU, or RNN).
        decoder (nn.Linear): Linear layer for output logits.
        rnn_type (str): Type of RNN used ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU').
        nhid (int): Number of hidden units in the RNN.
        nlayers (int): Number of layers in the RNN.

    Args:
        vocab_size (int): Size of the vocabulary.
        unit (int): Number of units in the embedding layer. Default is 650.
        nhid (Optional[int]): Number of hidden units in the RNN. Defaults to `unit`.
        nlayers (int): Number of layers in the RNN. Default is 2.
        dropout_rate (float): Dropout rate for regularization. Default is 0.0.
        tie_weights (bool): If True, tie the weights of the encoder and decoder.
        rnn_type (str): Type of RNN to use ('lstm', 'gru', 'rnn_tanh', 'rnn_relu').
        ignore_id (int): Padding index for the embedding layer. Default is 0.

    Raises:
        ValueError: If an invalid `rnn_type` is provided or if `nhid` does not
                     equal `unit` when `tie_weights` is True.

    Examples:
        # Create an instance of the SequentialRNNLM
        model = SequentialRNNLM(vocab_size=10000, unit=650, nlayers=2)

        # Initialize the hidden state
        hidden_state = model.zero_state()

        # Forward pass
        input_tensor = torch.randint(0, 10000, (32, 10))  # Batch of 32, seq len 10
        output, hidden_state = model(input_tensor, hidden_state)

        # Scoring a new token
        new_token = torch.tensor([5])  # Example token
        logp, new_state = model.score(new_token, hidden_state, input_tensor)

        # Batch scoring
        prefix_tokens = torch.randint(0, 10000, (32, 5))  # Batch of 32, prefix length 5
        scores, next_states = model.batch_score(prefix_tokens, [hidden_state]*32, input_tensor)
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        unit: int = 650,
        nhid: Optional[int] = None,
        nlayers: int = 2,
        dropout_rate: float = 0.0,
        tie_weights: bool = False,
        rnn_type: str = "lstm",
        ignore_id: int = 0,
    ):
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

    def zero_state(self):
        """
        Initialize LM state filled with zero values.

        This method creates an initial state for the language model (LM) that is
        filled with zeros. The shape of the state depends on the type of RNN being
        used. For LSTM networks, the state consists of two tensors representing
        the hidden state and the cell state. For other RNN types, it returns only
        the hidden state.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            A tuple containing the hidden and cell states for LSTM, or a single
            tensor for other RNN types. The shape of the returned tensor(s) is
            determined by the number of layers (`nlayers`) and the number of
            hidden units (`nhid`).

        Examples:
            >>> model = SequentialRNNLM(vocab_size=10000, unit=650, nlayers=2)
            >>> initial_state = model.zero_state()
            >>> initial_state
            (tensor([[0., 0., ..., 0.], [0., 0., ..., 0.]]),
             tensor([[0., 0., ..., 0.], [0., 0., ..., 0.]]))  # for LSTM

            >>> model = SequentialRNNLM(vocab_size=10000, unit=650, nlayers=1,
            ...                          rnn_type='RNN_TANH')
            >>> initial_state = model.zero_state()
            >>> initial_state
            tensor([[0., 0., ..., 0.]])  # for RNN
        """
        if isinstance(self.rnn, torch.nn.LSTM):
            h = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)
            c = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)
            state = h, c
        else:
            state = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)

        return state

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the RNN layer.

        This method computes the forward pass of the RNN language model. It
        takes the input tensor and the hidden state tensor, processes them
        through the embedding, RNN, and decoder layers, and returns the
        decoded output and the updated hidden state.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                containing token indices.
            hidden (torch.Tensor): Hidden state tensor of shape
                (num_layers, batch_size, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - torch.Tensor: Decoded output tensor of shape
                  (batch_size, seq_len, vocab_size).
                - torch.Tensor: Updated hidden state tensor.

        Examples:
            >>> model = SequentialRNNLM(vocab_size=1000, unit=650)
            >>> input_tensor = torch.randint(0, 1000, (32, 10))  # (batch_size, seq_len)
            >>> hidden_state = model.zero_state()  # Initialize hidden state
            >>> output, new_hidden = model.forward(input_tensor, hidden_state)
            >>> output.shape
            torch.Size([32, 10, 1000])  # (batch_size, seq_len, vocab_size)
        """
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

    def score(
        self,
        y: torch.Tensor,
        state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Score new token.

        This method computes the log probabilities for the next token based on
        the provided prefix tokens and the current state of the model.

        Args:
            y: 1D torch.int64 tensor representing the prefix tokens.
            state: The current scorer state for the prefix tokens, which can
                either be a tensor or a tuple of tensors (for LSTM).
            x: 2D torch.Tensor representing the encoder features that generate
                the tokens in `y`.

        Returns:
            Tuple of:
                - torch.float32 tensor containing the scores for the next token
                  (shape: n_vocab).
                - The updated state for the prefix tokens, which will be of the
                  same type as the input `state`.

        Examples:
            >>> model = SequentialRNNLM(vocab_size=1000)
            >>> prefix_tokens = torch.tensor([1, 2, 3])  # Example prefix tokens
            >>> initial_state = model.zero_state()  # Initialize state
            >>> encoder_features = torch.randn(1, 5, 650)  # Example features
            >>> scores, new_state = model.score(prefix_tokens, initial_state,
            ...                                   encoder_features)
        """
        y, new_state = self(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state

    def batch_score(
        self, ys: torch.Tensor, states: torch.Tensor, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        Raises:
            ValueError: If the state format is incorrect or if the model type
                is unsupported.

        Examples:
            >>> model = SequentialRNNLM(vocab_size=1000)
            >>> ys = torch.tensor([[1, 2], [3, 4]])  # Example prefix tokens
            >>> states = [model.zero_state() for _ in range(2)]  # Initial states
            >>> xs = torch.randn(2, 10, 300)  # Example encoder features
            >>> logp, new_states = model.batch_score(ys, states, xs)
            >>> print(logp.shape)  # Should print torch.Size([2, 1000])

        Note:
            This method processes a batch of input tokens and their corresponding
            states to produce the scores for the next possible tokens.
        """
        if states[0] is None:
            states = None
        elif isinstance(self.rnn, torch.nn.LSTM):
            # states: Batch x 2 x (Nlayers, Dim) -> 2 x (Nlayers, Batch, Dim)
            h = torch.stack([h for h, c in states], dim=1)
            c = torch.stack([c for h, c in states], dim=1)
            states = h, c
        else:
            # states: Batch x (Nlayers, Dim) -> (Nlayers, Batch, Dim)
            states = torch.stack(states, dim=1)

        ys, states = self(ys[:, -1:], states)
        # ys: (Batch, 1, Nvocab) -> (Batch, NVocab)
        assert ys.size(1) == 1, ys.shape
        ys = ys.squeeze(1)
        logp = ys.log_softmax(dim=-1)

        # state: Change to batch first
        if isinstance(self.rnn, torch.nn.LSTM):
            # h, c: (Nlayers, Batch, Dim)
            h, c = states
            # states: Batch x 2 x (Nlayers, Dim)
            states = [(h[:, i], c[:, i]) for i in range(h.size(1))]
        else:
            # states: (Nlayers, Batch, Dim) -> Batch x (Nlayers, Dim)
            states = [states[:, i] for i in range(states.size(1))]

        return logp, states
