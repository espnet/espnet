from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class RNNEncoderMix(AbsEncoder):
    """RNNEncoderMix class for multi-speaker mixture speech.

    Args:
        input_size: The number of expected features in the input
        output_size: The number of output features
        hidden_size: The number of hidden features
        bidirectional: If ``True`` becomes a bidirectional LSTM
        use_projection: Use projection layer or not
        num_layers_sd: Number of speaker-differentiating recurrent layers
        num_layers_rec: Number of shared recognition recurrent layers
        num_spkrs: Number of speakers
        dropout: dropout probability

    """

    def __init__(
        self,
        input_size: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        use_projection: bool = True,
        num_layers_sd: int = 4,
        num_layers_rec: int = 4,
        num_spkrs: int = 2,
        hidden_size: int = 320,
        output_size: int = 320,
        dropout: float = 0.0,
        subsample: Optional[Sequence[int]] = (2, 2, 1, 1),
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_projection = use_projection
        self.num_spkrs = num_spkrs

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        if subsample is None:
            subsample = np.ones(num_layers_sd + num_layers_rec + 1, dtype=np.int)
        else:
            subsample = subsample[: num_layers_sd + num_layers_rec]
            # Append 1 at the beginning because the second or later is used
            subsample = np.pad(
                np.array(subsample, dtype=np.int),
                [1, num_layers_sd + num_layers_rec - len(subsample)],
                mode="constant",
                constant_values=1,
            )

        rnn_type = ("b" if bidirectional else "") + rnn_type
        if use_projection:
            self.enc_sd = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            RNNP(
                                input_size,
                                num_layers_sd,
                                hidden_size,
                                output_size,
                                subsample[: num_layers_sd + 1],
                                dropout,
                                typ=rnn_type,
                            )
                        ]
                    )
                    for i in range(num_spkrs)
                ]
            )
            self.enc_rec = torch.nn.ModuleList(
                [
                    RNNP(
                        output_size,
                        num_layers_rec,
                        hidden_size,
                        output_size,
                        subsample[num_layers_rec:],
                        dropout,
                        typ=rnn_type,
                    )
                ]
            )

        else:
            self.enc_sd = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            RNN(
                                input_size,
                                num_layers_sd,
                                hidden_size,
                                output_size,
                                dropout,
                                typ=rnn_type,
                            )
                        ]
                    )
                    for i in range(num_spkrs)
                ]
            )
            self.enc_rec = torch.nn.ModuleList(
                [
                    RNN(
                        output_size,
                        num_layers_rec,
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
        # if prev_states is None:
        #     prev_states = [None] * len(self.enc)
        # assert len(prev_states) == len(self.enc)

        # current_states = []
        # for module, prev_state in zip(self.enc, prev_states):
        #     xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
        #     current_states.append(states)

        # SD and Rec encoder
        xs_pad_sd = [xs_pad for i in range(self.num_spkrs)]
        ilens_sd = [ilens for i in range(self.num_spkrs)]
        for ns in range(self.num_spkrs):
            # Encoder_SD: speaker differentiate encoder
            for module in self.enc_sd[ns]:
                xs_pad_sd[ns], ilens_sd[ns], _ = module(xs_pad_sd[ns], ilens_sd[ns])
            # Encoder_Rec: recognition encoder
            for module in self.enc_rec:
                xs_pad_sd[ns], ilens_sd[ns], _ = module(xs_pad_sd[ns], ilens_sd[ns])

        if self.use_projection:
            xs_pad.masked_fill_(make_pad_mask(ilens, xs_pad, 1), 0.0)
        else:
            xs_pad = xs_pad.masked_fill(make_pad_mask(ilens, xs_pad, 1), 0.0)
        if self.use_projection:
            for ns in range(self.num_spkrs):
                xs_pad_sd[ns].masked_fill_(
                    make_pad_mask(ilens_sd[ns], xs_pad_sd[ns], 1), 0.0
                )
        else:
            for ns in range(self.num_spkrs):
                xs_pad_sd[ns] = xs_pad_sd[ns].masked_fill(
                    make_pad_mask(ilens_sd[ns], xs_pad_sd[ns], 1), 0.0
                )
        return xs_pad_sd, ilens_sd, None
