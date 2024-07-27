from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN, RNNP


class RNNEncoder(AbsEncoder):
    """
        RNN-based encoder for speech recognition tasks.

    This class implements a recurrent neural network (RNN) encoder, which can be
    used as part of a speech recognition system. It supports both LSTM and GRU
    cell types, with options for bidirectionality and projection layers.

    Attributes:
        _output_size (int): The size of the output features.
        rnn_type (str): The type of RNN cell used ('lstm' or 'gru').
        bidirectional (bool): Whether the RNN is bidirectional.
        use_projection (bool): Whether to use projection layers.
        enc (torch.nn.ModuleList): List containing the RNN module(s).

    Args:
        input_size (int): The number of expected features in the input.
        rnn_type (str, optional): The type of RNN cell to use. Defaults to "lstm".
        bidirectional (bool, optional): If True, becomes a bidirectional RNN. Defaults to True.
        use_projection (bool, optional): Whether to use projection layers. Defaults to True.
        num_layers (int, optional): Number of recurrent layers. Defaults to 4.
        hidden_size (int, optional): The number of features in the hidden state. Defaults to 320.
        output_size (int, optional): The number of output features. Defaults to 320.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        subsample (Optional[Sequence[int]], optional): Subsampling factors for each layer.
            Defaults to (2, 2, 1, 1).

    Raises:
        ValueError: If an unsupported rnn_type is provided.

    Example:
        >>> input_size = 80
        >>> encoder = RNNEncoder(input_size, rnn_type="lstm", num_layers=3, hidden_size=256)
        >>> input_tensor = torch.randn(32, 100, input_size)  # (batch_size, sequence_length, input_size)
        >>> input_lengths = torch.randint(50, 100, (32,))  # (batch_size,)
        >>> output, output_lengths, _ = encoder(input_tensor, input_lengths)
        >>> print(output.shape)  # Expected: torch.Size([32, 25, 320])

    Note:
        The actual output sequence length may be shorter than the input due to subsampling.
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
                Get the output size of the encoder.

        Returns:
            int: The size of the output features from the encoder.

        Example:
            >>> encoder = RNNEncoder(input_size=80, output_size=512)
            >>> print(encoder.output_size())
            512
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
                Forward pass of the RNN encoder.

        This method processes the input tensor through the RNN layers, applying
        subsampling and masking as configured.

        Args:
            xs_pad (torch.Tensor): Padded input tensor of shape (batch, time, feat).
            ilens (torch.Tensor): Input lengths of each sequence in the batch.
            prev_states (torch.Tensor, optional): Previous hidden states for incremental processing.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - xs_pad (torch.Tensor): Output tensor after encoding and masking.
                - ilens (torch.Tensor): Output lengths of each sequence in the batch.
                - current_states (List[torch.Tensor]): List of current hidden states of the RNN.

        Example:
            >>> encoder = RNNEncoder(input_size=80, hidden_size=320, output_size=320)
            >>> xs_pad = torch.randn(2, 100, 80)  # (batch_size, time, feat)
            >>> ilens = torch.tensor([100, 80])
            >>> output, out_lens, states = encoder.forward(xs_pad, ilens)
            >>> print(output.shape)  # Expected: torch.Size([2, 25, 320])
            >>> print(out_lens)  # Expected: tensor([25, 20])

        Note:
            The output tensor is masked based on the output lengths to ensure
            padded regions are set to zero.
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
