from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN, RNNP


class RNNEncoder(AbsEncoder):
    """RNNEncoder class.

    Args:
        input_size: The number of expected features in the input
        output_size: The number of output features
        hidden_size: The number of hidden features
        bidirectional: If ``True`` becomes a bidirectional LSTM
        use_projection: Use projection layer or not
        num_layers: Number of recurrent layers
        dropout: dropout probability

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
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
