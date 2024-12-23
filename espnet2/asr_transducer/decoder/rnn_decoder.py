"""RNN decoder definition for Transducer models."""

from typing import List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder


class RNNDecoder(AbsDecoder):
    """
    RNN decoder definition for Transducer models.

    This class implements an RNN decoder module used in Transducer models. It
    supports both LSTM and GRU architectures and allows for customization of
    various parameters such as embedding size, hidden size, and dropout rates.

    Attributes:
        embed (torch.nn.Embedding): Embedding layer for the input labels.
        dropout_embed (torch.nn.Dropout): Dropout layer for the embedding.
        rnn (torch.nn.ModuleList): List of RNN layers (LSTM/GRU).
        dropout_rnn (torch.nn.ModuleList): List of dropout layers for RNN outputs.
        dlayers (int): Number of decoder layers.
        dtype (str): Type of RNN used ('lstm' or 'gru').
        output_size (int): Size of the output from the decoder.
        vocab_size (int): Size of the vocabulary.
        device (torch.device): Device to run the model on (CPU/GPU).
        score_cache (dict): Cache for storing scores of previous hypotheses.

    Args:
        vocab_size (int): Vocabulary size.
        embed_size (int, optional): Embedding size. Default is 256.
        hidden_size (int, optional): Hidden size. Default is 256.
        rnn_type (str, optional): Decoder layers type ('lstm' or 'gru').
            Default is 'lstm'.
        num_layers (int, optional): Number of decoder layers. Default is 1.
        dropout_rate (float, optional): Dropout rate for decoder layers.
            Default is 0.0.
        embed_dropout_rate (float, optional): Dropout rate for embedding layer.
            Default is 0.0.
        embed_pad (int, optional): Embedding padding symbol ID. Default is 0.

    Examples:
        # Create an RNNDecoder instance
        decoder = RNNDecoder(vocab_size=1000, embed_size=256, hidden_size=512)

        # Forward pass with a batch of label sequences
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        output = decoder(labels)

        # Initialize decoder states
        states = decoder.init_state(batch_size=2)

        # One-step forward hypothesis scoring
        out, new_states = decoder.score(label_sequence=[1, 2], states=states)

    Note:
        The decoder supports only 'lstm' and 'gru' as valid RNN types.
        Attempting to use any other type will raise a ValueError.

    Todo:
        - Implement functionality for more RNN types if needed.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        embed_dropout_rate: float = 0.0,
        embed_pad: int = 0,
    ) -> None:
        """Construct a RNNDecoder object."""
        super().__init__()

        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=embed_dropout_rate)

        rnn_class = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU

        self.rnn = torch.nn.ModuleList(
            [rnn_class(embed_size, hidden_size, 1, batch_first=True)]
        )

        for _ in range(1, num_layers):
            self.rnn += [rnn_class(hidden_size, hidden_size, 1, batch_first=True)]

        self.dropout_rnn = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout_rate) for _ in range(num_layers)]
        )

        self.dlayers = num_layers
        self.dtype = rnn_type

        self.output_size = hidden_size
        self.vocab_size = vocab_size

        self.device = next(self.parameters()).device
        self.score_cache = {}

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        RNN decoder definition for Transducer models.

        This module implements an RNN-based decoder for use in Transducer models,
        utilizing either LSTM or GRU architectures. It is designed to process
        label sequences and output decoder states for further processing in
        sequence modeling tasks.

        Attributes:
            embed: Embedding layer for the input label sequences.
            dropout_embed: Dropout layer for the embedding output.
            rnn: List of RNN layers (LSTM or GRU).
            dropout_rnn: List of dropout layers for the RNN outputs.
            dlayers: Number of decoder layers.
            dtype: Type of RNN used ('lstm' or 'gru').
            output_size: Size of the output from the decoder.
            vocab_size: Size of the vocabulary.
            device: Device on which the model is stored (CPU or GPU).
            score_cache: Cache for storing previously computed scores.

        Args:
            vocab_size (int): Vocabulary size.
            embed_size (int, optional): Size of the embedding layer. Default is 256.
            hidden_size (int, optional): Size of the hidden layers. Default is 256.
            rnn_type (str, optional): Type of RNN layers ('lstm' or 'gru'). Default is 'lstm'.
            num_layers (int, optional): Number of decoder layers. Default is 1.
            dropout_rate (float, optional): Dropout rate for decoder layers. Default is 0.0.
            embed_dropout_rate (float, optional): Dropout rate for embedding layer.
                Default is 0.0.
            embed_pad (int, optional): Padding symbol ID for the embedding layer. Default is 0.

        Examples:
            # Initialize the RNNDecoder
            decoder = RNNDecoder(vocab_size=1000, embed_size=256, hidden_size=256)

            # Forward pass with labels
            labels = torch.randint(0, 1000, (32, 10))  # (B, L)
            output = decoder.forward(labels)  # (B, U, D_dec)

        Returns:
            out (torch.Tensor): Decoder output sequences of shape (B, U, D_dec).

        Raises:
            ValueError: If the specified rnn_type is not supported (not 'lstm' or 'gru').
        """
        states = self.init_state(labels.size(0))

        embed = self.dropout_embed(self.embed(labels))
        out, _ = self.rnn_forward(embed, states)

        return out

    def rnn_forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        RNN decoder definition for Transducer models.

        This module implements an RNN decoder for Transducer models, utilizing
        either LSTM or GRU architectures. The decoder processes input sequences
        and produces output sequences, making it suitable for applications in
        automatic speech recognition (ASR).

        Attributes:
            vocab_size: Size of the vocabulary.
            embed_size: Size of the embedding layer.
            hidden_size: Size of the hidden layers.
            dtype: Type of RNN used ('lstm' or 'gru').
            dlayers: Number of decoder layers.
            score_cache: Cache for storing computed scores for efficiency.

        Args:
            vocab_size (int): Vocabulary size.
            embed_size (int, optional): Embedding size. Default is 256.
            hidden_size (int, optional): Hidden size. Default is 256.
            rnn_type (str, optional): Type of RNN layers ('lstm' or 'gru'). Default is 'lstm'.
            num_layers (int, optional): Number of decoder layers. Default is 1.
            dropout_rate (float, optional): Dropout rate for decoder layers. Default is 0.0.
            embed_dropout_rate (float, optional): Dropout rate for embedding layer. Default is 0.0.
            embed_pad (int, optional): Embedding padding symbol ID. Default is 0.

        Examples:
            # Initialize RNNDecoder
            decoder = RNNDecoder(vocab_size=1000, embed_size=256, hidden_size=256)

            # Forward pass with label sequences
            labels = torch.randint(0, 1000, (32, 10))  # Batch of 32 sequences of length 10
            output = decoder(labels)

            # One-step scoring
            label_sequence = [1, 2, 3]
            states = decoder.init_state(batch_size=1)
            out, states = decoder.score(label_sequence, states)

        Note:
            The decoder supports both LSTM and GRU architectures, but the choice
            should be made based on the specific requirements of the task.

        Todo:
            - Add support for more RNN types.
            - Implement attention mechanisms for enhanced performance.
        """
        h_prev, c_prev = state
        h_next, c_next = self.init_state(x.size(0))

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                x, (h_next[layer : layer + 1], c_next[layer : layer + 1]) = self.rnn[
                    layer
                ](x, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1]))
            else:
                x, h_next[layer : layer + 1] = self.rnn[layer](
                    x, hx=h_prev[layer : layer + 1]
                )

            x = self.dropout_rnn[layer](x)

        return x, (h_next, c_next)

    def score(
        self,
        label_sequence: List[int],
        states: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        RNN decoder definition for Transducer models.

        This module implements an RNN-based decoder for Transducer models, allowing for
        sequence-to-sequence tasks. The decoder can utilize either LSTM or GRU cells
        and supports multiple layers and dropout for regularization.

        Attributes:
            embed: Embedding layer for input sequences.
            dropout_embed: Dropout layer for embeddings.
            rnn: List of RNN layers (LSTM or GRU).
            dropout_rnn: List of dropout layers for RNN outputs.
            dlayers: Number of decoder layers.
            dtype: Type of RNN used ('lstm' or 'gru').
            output_size: Size of the decoder output.
            vocab_size: Size of the vocabulary.
            device: Device (CPU or GPU) on which the model resides.
            score_cache: Cache for previously computed scores to avoid redundant
                calculations.

        Args:
            vocab_size (int): Vocabulary size.
            embed_size (int): Embedding size (default: 256).
            hidden_size (int): Hidden size (default: 256).
            rnn_type (str): Type of RNN layers ('lstm' or 'gru', default: 'lstm').
            num_layers (int): Number of decoder layers (default: 1).
            dropout_rate (float): Dropout rate for decoder layers (default: 0.0).
            embed_dropout_rate (float): Dropout rate for embedding layer (default: 0.0).
            embed_pad (int): Embedding padding symbol ID (default: 0).

        Examples:
            decoder = RNNDecoder(vocab_size=1000, embed_size=256, hidden_size=256)
            labels = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Example label sequences
            output = decoder.forward(labels)

        Raises:
            ValueError: If an unsupported rnn_type is provided during initialization.
        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            out, states = self.score_cache[str_labels]
        else:
            label = torch.full(
                (1, 1),
                label_sequence[-1],
                dtype=torch.long,
                device=self.device,
            )

            embed = self.embed(label)
            out, states = self.rnn_forward(embed, states)

            self.score_cache[str_labels] = (out, states)

        return out[0], states

    def batch_score(
        self,
        hyps: List[Hypothesis],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        One-step forward hypotheses.

        This method takes a list of hypotheses and computes the decoder output
        sequences for each hypothesis. It utilizes the last label from each
        hypothesis to generate the embeddings and feed them into the RNN.

        Args:
            hyps: A list of Hypothesis objects, each containing a sequence of
                label IDs and the corresponding decoder states.

        Returns:
            out: Decoder output sequences of shape (B, D_dec), where B is the
                batch size and D_dec is the decoder output dimension.
            states: Decoder hidden states in the form of a tuple containing
                two elements:
                    - The hidden states of shape ((N, B, D_dec), ...)
                    - The cell states (only present if using LSTM), also of
                      shape ((N, B, D_dec), ...).

        Examples:
            >>> from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
            >>> from espnet2.asr_transducer.beam_search_transducer import Hypothesis
            >>> decoder = RNNDecoder(vocab_size=100, embed_size=64, hidden_size=128)
            >>> hyps = [Hypothesis(yseq=[1, 2, 3], dec_state=None),
            ...          Hypothesis(yseq=[4, 5, 6], dec_state=None)]
            >>> out, states = decoder.batch_score(hyps)
            >>> print(out.shape)  # Output shape should be (2, 128)
        """
        labels = torch.tensor(
            [[h.yseq[-1]] for h in hyps], dtype=torch.long, device=self.device
        )
        embed = self.embed(labels)

        states = self.create_batch_states([h.dec_state for h in hyps])
        out, states = self.rnn_forward(embed, states)

        return out.squeeze(1), states

    def set_device(self, device: torch.device) -> None:
        """
        Set the GPU device to use for the RNN decoder.

        This method updates the device attribute of the RNNDecoder class,
        allowing the model to run on the specified device (CPU or GPU).

        Args:
            device: The device ID (torch.device) to be set for the model.

        Examples:
            >>> decoder = RNNDecoder(vocab_size=1000)
            >>> decoder.set_device(torch.device('cuda:0'))

        Note:
            The device should be a valid torch.device object, which can be
            created using `torch.device('cpu')` or `torch.device('cuda:0')`.
        """
        self.device = device

    def init_state(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """
        RNN decoder definition for Transducer models.

        This module implements an RNN decoder for use in transducer models, providing
        a mechanism to process sequences of data through recurrent neural networks.
        The decoder can be configured with various parameters to adjust its
        architecture and behavior.

        Attributes:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Size of the embeddings.
            hidden_size (int): Size of the hidden states.
            rnn_type (str): Type of RNN used ('lstm' or 'gru').
            num_layers (int): Number of layers in the decoder.
            dropout_rate (float): Dropout rate applied to decoder layers.
            embed_dropout_rate (float): Dropout rate applied to embedding layer.
            embed_pad (int): Padding symbol ID for embeddings.

        Args:
            vocab_size (int): Vocabulary size.
            embed_size (int, optional): Embedding size (default is 256).
            hidden_size (int, optional): Hidden size (default is 256).
            rnn_type (str, optional): Decoder layers type ('lstm' or 'gru',
                                    default is 'lstm').
            num_layers (int, optional): Number of decoder layers (default is 1).
            dropout_rate (float, optional): Dropout rate for decoder layers
                                            (default is 0.0).
            embed_dropout_rate (float, optional): Dropout rate for embedding layer
                                                (default is 0.0).
            embed_pad (int, optional): Embedding padding symbol ID (default is 0).

        Examples:
            >>> decoder = RNNDecoder(vocab_size=5000, embed_size=256, hidden_size=256)
            >>> input_tensor = torch.randint(0, 5000, (32, 10))  # Batch of 32, seq len 10
            >>> output = decoder(input_tensor)
            >>> output.shape
            torch.Size([32, 10, 256])  # (Batch, Sequence Length, Hidden Size)

        Raises:
            ValueError: If `rnn_type` is not 'lstm' or 'gru'.
        """
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.output_size,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.output_size,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

    def select_state(
        self, states: Tuple[torch.Tensor, Optional[torch.Tensor]], idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states.
                    ((N, B, D_dec), (N, B, D_dec) or None)
            idx: State ID to extract.

        Returns:
            Decoder hidden state for given ID.
            ((N, 1, D_dec), (N, 1, D_dec) or None)

        Examples:
            >>> decoder = RNNDecoder(vocab_size=10)
            >>> states = decoder.init_state(batch_size=2)
            >>> selected_state = decoder.select_state(states, idx=0)
            >>> print(selected_state)
            (tensor(...), tensor(...) or None)

        Note:
            The function assumes that the states are in the expected format.
        """
        return (
            states[0][:, idx : idx + 1, :],
            states[1][:, idx : idx + 1, :] if self.dtype == "lstm" else None,
        )

    def create_batch_states(
        self,
        new_states: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create decoder hidden states.

        Args:
            new_states: List of decoder hidden states, where each state is a tuple
                of tensors. Each tensor corresponds to a specific hypothesis, and the
                format is as follows:
                - For LSTM: (N, 1, D_dec)
                - For GRU: (N, 1, D_dec) or None

        Returns:
            states: Combined decoder hidden states. The output is a tuple of tensors
                structured as:
                - For LSTM: ((N, B, D_dec), (N, B, D_dec))
                - For GRU: ((N, B, D_dec), None)

        Examples:
            >>> new_states = [(torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)),
            ...               (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))]
            >>> batch_states = create_batch_states(new_states)
            >>> print(batch_states[0].shape)  # Output: (2, 2, 256)
            >>> print(batch_states[1].shape)  # Output: (2, 2, 256)
        """
        return (
            torch.cat([s[0] for s in new_states], dim=1),
            (
                torch.cat([s[1] for s in new_states], dim=1)
                if self.dtype == "lstm"
                else None
            ),
        )
