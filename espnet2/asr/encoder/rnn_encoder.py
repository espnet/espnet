from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN, RNNP


class RNNEncoder(AbsEncoder):
    """
    RNNEncoder class for sequence-to-sequence models using recurrent neural networks.

    This class implements an RNN-based encoder for processing sequential data, such 
    as speech signals in automatic speech recognition (ASR) tasks. It allows for 
    the use of either LSTM or GRU cells and supports bidirectional processing 
    and projection layers.

    Attributes:
        rnn_type (str): Type of RNN to use ('lstm' or 'gru').
        bidirectional (bool): If True, uses a bidirectional RNN.
        use_projection (bool): If True, applies a projection layer.
        _output_size (int): The number of output features.

    Args:
        input_size (int): The number of expected features in the input.
        rnn_type (str, optional): Type of RNN ('lstm' or 'gru'). Default is 'lstm'.
        bidirectional (bool, optional): Whether to use a bidirectional RNN. 
            Default is True.
        use_projection (bool, optional): Whether to use a projection layer. 
            Default is True.
        num_layers (int, optional): Number of recurrent layers. Default is 4.
        hidden_size (int, optional): Number of hidden features. Default is 320.
        output_size (int, optional): Number of output features. Default is 320.
        dropout (float, optional): Dropout probability. Default is 0.0.
        subsample (Sequence[int], optional): Subsampling factors for each layer. 
            Default is (2, 2, 1, 1).

    Raises:
        ValueError: If the provided rnn_type is not supported.

    Examples:
        # Initialize an RNNEncoder
        encoder = RNNEncoder(input_size=40, hidden_size=256, output_size=256)

        # Forward pass through the encoder
        xs_pad = torch.randn(10, 5, 40)  # (sequence_length, batch_size, input_size)
        ilens = torch.tensor([5, 4, 3, 5, 2])  # Actual lengths of sequences
        output, lengths, states = encoder(xs_pad, ilens)

    Note:
        This encoder can be easily integrated into larger ASR systems 
        and supports various configurations based on task requirements.

    Todo:
        - Implement additional functionalities such as saving and loading 
        model states.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        use_projection: bool = True,
        num_layers: int = 4,
        hidden_size: int = 320,
        output_size: int = 320,
        dropout: float = 0.0,
        subsample: Optional[Sequence[int]] = (2, 2, 1, 1),
    ):
        super().__init__()
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_projection = use_projection

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        if subsample is None:
            _subsample = np.ones(num_layers + 1, dtype=np.int64)
        else:
            _subsample = subsample[:num_layers]
            # Append 1 at the beginning because the second or later is used
            _subsample = np.pad(
                np.array(_subsample, dtype=np.int64),
                [1, num_layers - len(_subsample)],
                mode="constant",
                constant_values=1,
            )

        rnn_type = ("b" if bidirectional else "") + rnn_type
        if use_projection:
            self.enc = torch.nn.ModuleList(
                [
                    RNNP(
                        input_size,
                        num_layers,
                        hidden_size,
                        output_size,
                        _subsample,
                        dropout,
                        typ=rnn_type,
                    )
                ]
            )

        else:
            self.enc = torch.nn.ModuleList(
                [
                    RNN(
                        input_size,
                        num_layers,
                        hidden_size,
                        output_size,
                        dropout,
                        typ=rnn_type,
                    )
                ]
            )

    def output_size(self) -> int:
        """
        Return the size of the output features.

        This method retrieves the number of output features defined during the
        initialization of the RNNEncoder. The output size is crucial for
        subsequent layers in a neural network model, ensuring that the output
        dimensions match the expected input dimensions of any following
        layers or components.

        Returns:
            int: The number of output features defined in the RNNEncoder.

        Examples:
            >>> encoder = RNNEncoder(input_size=128, output_size=256)
            >>> encoder.output_size()
            256

        Note:
            This method is particularly useful when you need to understand
            the dimensions of the output from the RNN layer, especially when
            designing architectures that require precise input-output shape
            matching.

        Todo:
            Consider adding additional functionality to handle dynamic output
            sizes if needed in future implementations.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input sequences through the RNN encoder.

        This method takes padded input sequences along with their lengths and
        optionally previous states, and passes them through the RNN encoder.
        The output consists of the encoded features, updated input lengths,
        and the current states of the RNN.

        Args:
            xs_pad (torch.Tensor): Padded input sequences of shape (T, N, C),
                where T is the maximum sequence length, N is the batch size,
                and C is the number of input features.
            ilens (torch.Tensor): Lengths of each sequence in the batch
                of shape (N,).
            prev_states (torch.Tensor, optional): Previous states of the RNN,
                defaults to None, which initializes states to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - xs_pad (torch.Tensor): The processed padded input sequences
                  after passing through the RNN encoder.
                - ilens (torch.Tensor): Updated lengths of each sequence
                  after processing.
                - current_states (torch.Tensor): The updated states of the
                  RNN after processing.

        Examples:
            >>> encoder = RNNEncoder(input_size=10, output_size=20)
            >>> xs_pad = torch.rand(5, 3, 10)  # (T=5, N=3, C=10)
            >>> ilens = torch.tensor([5, 3, 4])  # Lengths of sequences
            >>> output, lengths, states = encoder.forward(xs_pad, ilens)

        Note:
            Ensure that `xs_pad` is padded correctly and that `ilens`
            corresponds to the actual lengths of the sequences in the batch.

        Raises:
            AssertionError: If the length of `prev_states` does not match
            the number of RNN layers.
        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        if self.use_projection:
            xs_pad.masked_fill_(make_pad_mask(ilens, xs_pad, 1), 0.0)
        else:
            xs_pad = xs_pad.masked_fill(make_pad_mask(ilens, xs_pad, 1), 0.0)
        return xs_pad, ilens, current_states
