import logging
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet.nets.pytorch_backend.rnn.encoders import VGG2L
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class VGGRNNEncoderMix(AbsEncoder):
    """VGGRNNEncoderMix class.

    Args:
        input_size: The number of expected features in the input
        bidirectional: If ``True`` becomes a bidirectional LSTM
        use_projection: Use projection layer or not
        num_layers_sd: Number of speaker-differentiating recurrent layers
        num_layers_rec: Number of shared recognition recurrent layers
        num_spkrs: Number of speakers
        hidden_size: The number of hidden features
        output_size: The number of output features
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
        in_channel: int = 1,
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

        # Subsample is not used for VGGRNN
        subsample = np.ones(num_layers_sd + num_layers_rec + 1, dtype=np.int)
        rnn_type = ("b" if bidirectional else "") + rnn_type
        self.enc_mix = torch.nn.ModuleList([VGG2L(in_channel)])
        if use_projection:
            self.enc_sd = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            RNNP(
                                get_vgg2l_odim(input_size, in_channel=in_channel),
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
            logging.info("Use CNN-VGG + B" + rnn_type.upper() + "P for encoder")

        else:
            self.enc_sd = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            RNN(
                                get_vgg2l_odim(input_size, in_channel=in_channel),
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
            logging.info("Use CNN-VGG + B" + rnn_type.upper() + " for encoder")

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

        # mixture encoder
        for module in self.enc_mix:
            xs_pad, ilens, _ = module(xs_pad, ilens)

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
