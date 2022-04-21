"""RNNP block for Transducer encoder."""

from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from typeguard import check_argument_types


class RNNP(torch.nn.Module):
    """RNNP module definition.

    Args:
        dim_input: Input dimension.
        dim_hidden: Hidden dimension.
        dim_proj: Projection dimension.
        rnn_type: Type of RNN layers.
        bidirectional: Whether bidirectional layers are used.
        num_blocks: Number of layers.
        subsample: Subsampling for each layer.
        dropout_rate: Dropout rate.
        dim_output: Output dimension if provided.

    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_proj: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_blocks: int = 1,
        dropout_rate: float = 0.0,
        subsample: Optional[Sequence[int]] = None,
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

        self.rnn = torch.nn.ModuleList(
            [
                rnn_class(
                    dim_input,
                    dim_hidden,
                    1,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            ]
        )
        self.lin_proj = torch.nn.ModuleList(
            [torch.nn.Linear(2 * dim_hidden if bidirectional else dim_hidden, dim_proj)]
        )
        self.dropout = torch.nn.ModuleList([])

        for _ in range(1, num_blocks):
            self.rnn += [
                rnn_class(
                    dim_proj,
                    dim_hidden,
                    1,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            ]
            self.lin_proj += [
                torch.nn.Linear(
                    2 * dim_hidden if bidirectional else dim_hidden, dim_proj
                )
            ]

            self.dropout += [torch.nn.Dropout(p=dropout_rate)]

        if subsample is None:
            self.subsample = np.ones(num_blocks + 1, dtype=int)
        else:
            subsample = subsample[:num_blocks]

            self.subsample = np.pad(
                np.array(subsample, dtype=int),
                [1, num_blocks - len(subsample)],
                mode="constant",
                constant_values=1,
            )

        self.num_blocks = num_blocks
        self.dim_hidden = dim_hidden

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_len: torch.Tensor,
        cache: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: RNNP input sequences. (B, T, D_emb)
            sequence_len: Input sequences lengths. (B,)

        Returns:
            sequence: RNNP output sequences. (B, T, D_enc)
            sequence_len: Output sequences lengths. (B,)

        """
        states = cache

        for block in range(self.num_blocks):
            sequence = pack_padded_sequence(
                sequence, sequence_len.cpu(), batch_first=True
            )

            if self.training:
                self.rnn[block].flatten_parameters()

            sequence, states = self.rnn[block](sequence, states)

            sequence, sequence_len = pad_packed_sequence(sequence, batch_first=True)

            sub = self.subsample[block + 1]
            if sub > 1:
                sequence = sequence[:, ::sub]
                sequence_len = torch.tensor([int(i + 1) // sub for i in sequence_len])

            sequence_proj = self.lin_proj[block](
                sequence.contiguous().view(-1, sequence.size(2))
            )
            sequence = sequence_proj.view(sequence.size(0), sequence.size(1), -1)

            if block < self.num_blocks - 1:
                sequence = self.dropout[block](sequence)

        return sequence, sequence_len
