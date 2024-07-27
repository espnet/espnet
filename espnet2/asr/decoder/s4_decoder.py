"""Decoder definition."""

from typing import Any, List, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.state_spaces.model import SequenceModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.scorer_interface import BatchScorerInterface


class S4Decoder(AbsDecoder, BatchScorerInterface):
    """
        S4 decoder module for sequence-to-sequence models.

    This class implements a decoder based on the S4 (Structured State Space Sequence) model.
    It can be used in various sequence-to-sequence tasks, such as automatic speech recognition.

    Args:
        vocab_size (int): Size of the vocabulary (output dimension).
        encoder_output_size (int): Dimension of the encoder's hidden vector.
        input_layer (str, optional): Type of input layer. Defaults to "embed".
        dropinp (float, optional): Input dropout rate. Defaults to 0.0.
        dropout (float, optional): Dropout rate applied to every residual and layer. Defaults to 0.25.
        prenorm (bool, optional): If True, use pre-normalization; otherwise, post-normalization. Defaults to True.
        n_layers (int, optional): Number of layers in the decoder. Defaults to 16.
        transposed (bool, optional): If True, transpose inputs so each layer receives (batch, dim, length). Defaults to False.
        tie_dropout (bool, optional): If True, tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d. Defaults to False.
        n_repeat (int, optional): Number of times each layer is repeated per stage before applying pooling. Defaults to 1.
        layer (dict, optional): Layer configuration. Must be specified.
        residual (dict, optional): Residual connection configuration.
        norm (dict, optional): Normalization configuration (e.g., layer vs batch).
        pool (dict, optional): Configuration for pooling layer per stage.
        track_norms (bool, optional): If True, log norms of each layer output. Defaults to True.
        drop_path (float, optional): Drop rate for stochastic depth. Defaults to 0.0.

    Attributes:
        d_model (int): Dimension of the model (same as encoder_output_size).
        sos (int): Start-of-sequence token ID.
        eos (int): End-of-sequence token ID.
        odim (int): Output dimension (same as vocab_size).
        dropout (float): Dropout rate.
        embed (torch.nn.Embedding): Embedding layer for input tokens.
        dropout_emb (torch.nn.Dropout): Dropout layer for embeddings.
        decoder (SequenceModel): Main S4 decoder model.
        output (torch.nn.Linear): Output linear layer.

    Note:
        This decoder implements the BatchScorerInterface, allowing for efficient batch scoring of hypotheses.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        input_layer: str = "embed",
        dropinp: float = 0.0,
        dropout: float = 0.25,
        prenorm: bool = True,
        n_layers: int = 16,
        transposed: bool = False,
        tie_dropout: bool = False,
        n_repeat=1,
        layer=None,
        residual=None,
        norm=None,
        pool=None,
        track_norms=True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.d_model = encoder_output_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.odim = vocab_size
        self.dropout = dropout

        if input_layer == "embed":
            self.embed = torch.nn.Embedding(vocab_size, self.d_model)
        else:
            raise NotImplementedError
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = SequenceModel(
            self.d_model,
            n_layers=n_layers,
            transposed=transposed,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            n_repeat=n_repeat,
            layer=layer,
            residual=residual,
            norm=norm,
            pool=pool,
            track_norms=track_norms,
            dropinp=dropinp,
            drop_path=drop_path,
        )

        self.output = torch.nn.Linear(self.d_model, vocab_size)

    def init_state(self, x: torch.Tensor):
        """
                Initialize the decoder state.

        This method initializes the state of the S4 decoder. It creates a default state
        for the decoder based on the input tensor's device.

        Args:
            x (torch.Tensor): An input tensor used to determine the device for state initialization.

        Returns:
            Any: The initialized state of the decoder.

        Note:
            The state is initialized with a batch size of 1, regardless of the input tensor's batch size.
            This method is typically called before starting the decoding process.

        Example:
            >>> decoder = S4Decoder(vocab_size=1000, encoder_output_size=512)
            >>> x = torch.randn(1, 10, 512)  # (batch_size, sequence_length, feature_dim)
            >>> initial_state = decoder.init_state(x)
        """
        return self.decoder.default_state(1, device=x.device)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass of the S4 decoder.

        This method performs the forward pass of the S4 decoder, processing the encoder output
        and the input token sequence to generate decoded token scores.

        Args:
            hs_pad (torch.Tensor): Encoded memory, float32 tensor of shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of the encoded sequences in the batch, shape (batch,).
            ys_in_pad (torch.Tensor): Input token ids, int64 tensor of shape (batch, maxlen_out).
                If input_layer is "embed", this contains token ids.
                Otherwise, it contains input tensors of shape (batch, maxlen_out, #mels).
            ys_in_lens (torch.Tensor): Lengths of the input sequences in the batch, shape (batch,).
            state (Optional[Any]): Initial state for the decoder. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - decoded (torch.Tensor): Decoded token scores before softmax,
                  shape (batch, maxlen_out, vocab_size).
                - ys_in_lens (torch.Tensor): Lengths of the input sequences, shape (batch,).

        Note:
            This method applies the embedding layer, processes the sequence through the S4 decoder,
            and applies the output layer to generate token scores.

        Example:
            >>> decoder = S4Decoder(vocab_size=1000, encoder_output_size=512)
            >>> hs_pad = torch.randn(2, 100, 512)  # (batch_size, max_encoder_length, encoder_dim)
            >>> hlens = torch.tensor([100, 80])
            >>> ys_in_pad = torch.randint(0, 1000, (2, 20))  # (batch_size, max_decoder_length)
            >>> ys_in_lens = torch.tensor([20, 15])
            >>> decoded, out_lens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)
            >>> print(decoded.shape)  # (2, 20, 1000)
        """
        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        emb = self.embed(ys_in_pad)
        z, state = self.decoder(
            emb,
            state=state,
            memory=memory,
            lengths=ys_in_lens,
            mask=memory_mask,
        )

        decoded = self.output(z)
        return decoded, ys_in_lens

    def score(self, ys, state, x):
        """
                Score a sequence of tokens.

        This method is not implemented for the S4Decoder.

        Args:
            ys: The sequence of tokens to be scored.
            state: The current state of the decoder.
            x: The input features.

        Raises:
            NotImplementedError: This method is not implemented for S4Decoder.

        Note:
            This method is part of the BatchScorerInterface, but it is not implemented
            for the S4Decoder. Use the `batch_score` method instead for efficient
            batch scoring of hypotheses.
        """
        raise NotImplementedError

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
                Score new token batches efficiently.

        This method computes scores for the next token given a batch of prefix tokens
        and their corresponding states. It is part of the BatchScorerInterface and is
        used for efficient batch decoding.

        Args:
            ys (torch.Tensor): Prefix tokens of shape (n_batch, ylen).
                Contains int64 token IDs.
            states (List[Any]): List of scorer states for prefix tokens.
                Each state corresponds to a sequence in the batch.
            xs (torch.Tensor): The encoder features that generate ys.
                Shape (n_batch, xlen, n_feat).

        Returns:
            Tuple[torch.Tensor, List[Any]]: A tuple containing:
                - logp (torch.Tensor): Log probabilities for the next token.
                  Shape (n_batch, n_vocab).
                - states_list (List[Any]): Updated states for each sequence in the batch.

        Note:
            This method is designed for use in beam search decoding, where multiple
            hypotheses are scored simultaneously.

        Example:
            >>> decoder = S4Decoder(vocab_size=1000, encoder_output_size=512)
            >>> ys = torch.randint(0, 1000, (5, 10))  # (n_batch, ylen)
            >>> states = [decoder.init_state(torch.randn(1, 512)) for _ in range(5)]
            >>> xs = torch.randn(5, 100, 512)  # (n_batch, xlen, n_feat)
            >>> logp, new_states = decoder.batch_score(ys, states, xs)
            >>> print(logp.shape)  # (5, 1000)
        """
        # merge states
        n_batch = len(ys)
        ys = self.embed(ys[:, -1:])

        # workaround for remaining beam width of 1
        if type(states[0]) is list:
            states = states[0]

        assert ys.size(1) == 1, ys.shape
        ys = ys.squeeze(1)

        ys, states = self.decoder.step(ys, state=states, memory=xs)
        logp = self.output(ys).log_softmax(dim=-1)

        states_list = [
            [state[b].unsqueeze(0) if state is not None else None for state in states]
            for b in range(n_batch)
        ]

        return logp, states_list
