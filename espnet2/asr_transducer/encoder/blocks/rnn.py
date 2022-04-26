"""RNN block for Transducer encoder."""

from typing import Optional
from typing import Tuple

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from typeguard import check_argument_types


class RNN(torch.nn.Module):
    """RNN module definition.

    Args:
        dim_input: Input dimension.
        dim_hidden: Hidden dimension.
        rnn_type: Type of RNN layers.
        bidirectional: Whether bidirectional layers are used.
        num_blocks: Number of layers.
        dropout_rate: Dropout_Rate rate.
        dim_output: Output dimension if provided.

    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_blocks: int = 1,
        dropout_rate: float = 0.0,
        dim_output: Optional[int] = None,
    ):
        assert check_argument_types()

        super().__init__()

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")
        else:
            rnn_class = torch.nn.LSTM if "lstm" in rnn_type else torch.nn.GRU

        self.rnn = rnn_class(
            dim_input,
            dim_hidden,
            num_layers=num_blocks,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )

        if dim_output is None:
            dim_output = dim_hidden

        if bidirectional:
            self.output_layer = torch.nn.Linear(dim_hidden * 2, dim_output)
        else:
            self.output_layer = torch.nn.Linear(dim_hidden, dim_output)

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_len: torch.Tensor,
        cache: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: RNN input sequences. (B, T, D_emb)
            sequence_len: Input sequences lengths. (B,)

        Returns:
            sequence: RNN output sequences. (B, T, D_enc)
            sequence_len: Output sequences lengths. (B,)

        """
        sequence = pack_padded_sequence(sequence, sequence_len.cpu(), batch_first=True)

        if self.training:
            self.rnn.flatten_parameters()

        sequence, _ = self.rnn(sequence)

        sequence_pad, sequence_len = pad_packed_sequence(sequence, batch_first=True)

        sequence = torch.tanh(
            self.output_layer(sequence_pad.contiguous().view(-1, sequence_pad.size(2)))
        ).view(sequence_pad.size(0), sequence_pad.size(1), -1)

        return sequence, sequence_len
