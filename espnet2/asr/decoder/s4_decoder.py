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
    S4 decoder module for sequence-to-sequence tasks.

    This class implements the S4 decoder, which is a part of the ESPnet2 ASR
    (Automatic Speech Recognition) framework. The decoder processes the
    encoded input and generates token scores for the output sequence.

    Attributes:
        d_model (int): Dimension of the hidden state.
        sos (int): Start-of-sequence token ID.
        eos (int): End-of-sequence token ID.
        odim (int): Output dimension (vocabulary size).
        dropout (float): Dropout rate for regularization.
        embed (torch.nn.Embedding): Embedding layer for input tokens.
        dropout_emb (torch.nn.Dropout): Dropout layer for embeddings.
        decoder (SequenceModel): Sequence model implementing the core
            decoding functionality.
        output (torch.nn.Linear): Linear layer to project decoder outputs
            to vocabulary size.

    Args:
        vocab_size (int): Size of the output vocabulary.
        encoder_output_size (int): Dimension of the hidden vector from the
            encoder.
        input_layer (str): Type of input layer (default is "embed").
        dropinp (float): Input dropout rate (default is 0.0).
        dropout (float): Dropout parameter applied on every residual and
            every layer (default is 0.25).
        prenorm (bool): Flag for using pre-norm vs. post-norm (default is True).
        n_layers (int): Number of layers in the decoder (default is 16).
        transposed (bool): If True, transposes inputs for each layer
            (default is False).
        tie_dropout (bool): If True, ties dropout mask across sequences
            (default is False).
        n_repeat (int): Number of repetitions of each layer per stage
            (default is 1).
        layer (Any): Configuration for layers, must be specified.
        residual (Any): Configuration for residual connections.
        norm (Any): Normalization configuration (e.g., layer vs batch).
        pool (Any): Configuration for pooling layer per stage.
        track_norms (bool): If True, logs norms of each layer output
            (default is True).
        drop_path (float): Drop rate for stochastic depth (default is 0.0).

    Methods:
        init_state(x: torch.Tensor) -> Any:
            Initializes the state for the decoder based on the input tensor.

        forward(
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            state: Any = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            Processes the input through the decoder and returns the
            decoded token scores and output lengths.

        score(ys: torch.Tensor, state: Any, x: torch.Tensor) -> None:
            Computes the score for the new token (Not implemented).

        batch_score(
            ys: torch.Tensor,
            states: List[Any],
            xs: torch.Tensor
        ) -> Tuple[torch.Tensor, List[Any]]:
            Scores a batch of new tokens based on the current states and
            encoder features.

    Examples:
        decoder = S4Decoder(vocab_size=100, encoder_output_size=512)
        hs_pad = torch.randn(32, 50, 512)  # (batch, maxlen_in, feat)
        hlens = torch.randint(1, 51, (32,))  # (batch)
        ys_in_pad = torch.randint(0, 100, (32, 20))  # (batch, maxlen_out)
        ys_in_lens = torch.randint(1, 21, (32,))  # (batch)

        # Forward pass
        decoded_scores, output_lengths = decoder.forward(
            hs_pad, hlens, ys_in_pad, ys_in_lens
        )
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

        This method initializes the decoder's internal state using the
        specified input tensor. The state is required for processing the
        input sequences through the decoder.

        Args:
            x (torch.Tensor): Input tensor used to determine the device
                for the state initialization. The tensor shape should
                be compatible with the model's expected input.

        Returns:
            torch.Tensor: The initialized state of the decoder, which is
                a default state tensor suitable for starting the decoding
                process.

        Examples:
            >>> decoder = S4Decoder(vocab_size=100, encoder_output_size=512)
            >>> input_tensor = torch.randn(1, 10, 512)  # Example input
            >>> state = decoder.init_state(input_tensor)
            >>> state.shape
            torch.Size([1, <state_dimension>])  # Replace <state_dimension>
                                                  # with the actual dimension

        Note:
            The returned state is typically passed to the `forward` method
            during decoding.
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
        Forward decoder.

        This method processes the input through the decoder, utilizing the encoded
        memory and returning the decoded token scores along with the lengths of
        the output sequences.

        Args:
            hs_pad (torch.Tensor): Encoded memory, shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of the encoded sequences, shape (batch,).
            ys_in_pad (torch.Tensor): Input token IDs, shape (batch, maxlen_out).
                If `input_layer` is "embed", it contains token IDs; otherwise, it
                contains the input tensor in a different format.
            ys_in_lens (torch.Tensor): Lengths of the input sequences, shape (batch,).
            state (Any, optional): The state of the decoder. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax,
                  shape (batch, maxlen_out, vocab_size) if `use_output_layer`
                  is True.
                - olens (torch.Tensor): Lengths of the output sequences, shape (batch,).

        Examples:
            >>> decoder = S4Decoder(vocab_size=1000, encoder_output_size=512)
            >>> hs_pad = torch.rand(32, 10, 512)  # Example encoded memory
            >>> hlens = torch.randint(1, 11, (32,))  # Random lengths
            >>> ys_in_pad = torch.randint(0, 1000, (32, 20))  # Random token IDs
            >>> ys_in_lens = torch.randint(1, 21, (32,))  # Random lengths
            >>> output, output_lengths = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)

        Note:
            Ensure that the input tensors are properly padded and that the lengths
            provided correspond to the actual lengths of the sequences.

        Raises:
            ValueError: If the dimensions of the input tensors do not match the
            expected shapes.
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
        S4 decoder module for sequence-to-sequence tasks.

        This class implements an S4 decoder that can be used for various
        sequence generation tasks in neural networks. It leverages a
        sequence model and provides methods for both forward decoding
        and scoring new token batches.

        Args:
            vocab_size (int): Output dimension, representing the size of the
                vocabulary.
            encoder_output_size (int): Dimension of the hidden vector from the
                encoder.
            input_layer (str, optional): Type of input layer. Defaults to
                "embed".
            dropinp (float, optional): Dropout applied to the input layer.
                Defaults to 0.0.
            dropout (float, optional): Dropout parameter applied on every
                residual and every layer. Defaults to 0.25.
            prenorm (bool, optional): Whether to use pre-norm or post-norm.
                Defaults to True.
            n_layers (int, optional): Number of layers in the decoder.
                Defaults to 16.
            transposed (bool, optional): If True, transpose inputs so each
                layer receives (batch, dim, length). Defaults to False.
            tie_dropout (bool, optional): If True, tie dropout mask across
                sequences like nn.Dropout1d/nn.Dropout2d. Defaults to False.
            n_repeat (int, optional): Number of times each layer is repeated
                per stage before applying pooling. Defaults to 1.
            layer (optional): Layer configuration, must be specified.
            residual (optional): Residual configuration.
            norm (optional): Normalization configuration (e.g. layer vs batch).
            pool (optional): Configuration for pooling layer per stage.
            track_norms (bool, optional): If True, log norms of each layer
                output. Defaults to True.
            drop_path (float, optional): Drop rate for stochastic depth.
                Defaults to 0.0.

        Attributes:
            d_model (int): The dimension of the model.
            sos (int): Start of sequence token index.
            eos (int): End of sequence token index.
            odim (int): Output dimension (vocab size).
            dropout (float): Dropout rate.

        Methods:
            init_state(x: torch.Tensor) -> Any:
                Initializes the state for the decoder.
            forward(hs_pad: torch.Tensor, hlens: torch.Tensor,
                    ys_in_pad: torch.Tensor, ys_in_lens: torch.Tensor,
                    state=None) -> Tuple[torch.Tensor, torch.Tensor]:
                Performs a forward pass through the decoder.
            score(ys: torch.Tensor, state: Any, x: torch.Tensor):
                Calculates the score for the given input.
            batch_score(ys: torch.Tensor, states: List[Any],
                        xs: torch.Tensor) -> Tuple[torch.Tensor, List[Any]]:
                Scores a batch of new tokens.

        Raises:
            NotImplementedError: If the input layer type is not supported or
                if the score method is called.

        Examples:
            decoder = S4Decoder(vocab_size=100, encoder_output_size=256)
            state = decoder.init_state(torch.zeros(1, 256))
            output, lengths = decoder.forward(
                hs_pad=torch.randn(32, 10, 256),
                hlens=torch.tensor([10]*32),
                ys_in_pad=torch.randint(0, 100, (32, 20)),
                ys_in_lens=torch.tensor([20]*32),
                state=state
            )
            print(output.shape)  # Output: (32, 20, 100)

            # Scoring a batch of tokens
            scores, new_states = decoder.batch_score(
                ys=torch.randint(0, 100, (32, 1)),
                states=[state]*32,
                xs=torch.randn(32, 10, 256)
            )
            print(scores.shape)  # Output: (32, 100)
        """
        raise NotImplementedError

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
        Score new token batch.

        This method computes the scores for the next token in a batch of sequences
        given the prefix tokens and their corresponding states. It utilizes the
        decoder's embedding layer and processes the input through the decoder to
        generate the scores for the next token in the vocabulary.

        Args:
            ys (torch.Tensor): A tensor of shape (n_batch, ylen) containing the
                prefix tokens of type torch.int64.
            states (List[Any]): A list of states associated with the prefix tokens,
                which are used for scoring.
            xs (torch.Tensor): A tensor of shape (n_batch, xlen, n_feat) that
                contains the encoder features corresponding to the prefix tokens.

        Returns:
            Tuple[torch.Tensor, List[Any]]: A tuple containing:
                - A tensor of shape (n_batch, n_vocab) representing the
                  batchified scores for the next token.
                - A list of next state lists for each prefix token in `ys`.

        Examples:
            >>> decoder = S4Decoder(vocab_size=100, encoder_output_size=64)
            >>> ys = torch.tensor([[1, 2, 3], [2, 3, 4]])
            >>> states = [None, None]  # Example states
            >>> xs = torch.randn(2, 10, 64)  # Example encoder features
            >>> scores, next_states = decoder.batch_score(ys, states, xs)
            >>> print(scores.shape)  # Should print torch.Size([2, 100])

        Note:
            Ensure that the last token in `ys` is used for scoring the next token.
            This function is designed for batch processing of tokens, and the
            implementation assumes that the states are correctly managed across
            the batch.

        Raises:
            NotImplementedError: If the method is not fully implemented or if
            unsupported state types are provided.
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
