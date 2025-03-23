"""RNN encoder implementation for Transducer model.

These classes are based on the ones in espnet.nets.pytorch_backend.rnn.encoders,
and modified to output intermediate representation based given list of layers as input.
To do so, RNN class rely on a stack of 1-layer LSTM instead of a multi-layer LSTM.
The additional outputs are intended to be used with Transducer auxiliary tasks.


"""

from argparse import Namespace
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, to_device


class RNNP(torch.nn.Module):
    """RNN with projection layer module.

    Args:
        idim: Input dimension.
        rnn_type: RNNP units type.
        elayers: Number of RNNP layers.
        eunits: Number of units ((2 * eunits) if bidirectional).
        eprojs: Number of projection units.
        subsample: Subsampling rate per layer.
        dropout_rate: Dropout rate for RNNP layers.
        aux_output_layers: Layer IDs for auxiliary RNNP output sequences.

    """

    def __init__(
        self,
        idim: int,
        rnn_type: str,
        elayers: int,
        eunits: int,
        eprojs: int,
        subsample: np.ndarray,
        dropout_rate: float,
        aux_output_layers: List = [],
    ):
        """Initialize RNNP module."""
        super().__init__()

        bidir = rnn_type[0] == "b"
        for i in range(elayers):
            if i == 0:
                input_dim = idim
            else:
                input_dim = eprojs

            rnn_layer = torch.nn.LSTM if "lstm" in rnn_type else torch.nn.GRU
            rnn = rnn_layer(
                input_dim, eunits, num_layers=1, bidirectional=bidir, batch_first=True
            )

            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)

            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * eunits, eprojs))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(eunits, eprojs))

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.elayers = elayers
        self.eunits = eunits
        self.subsample = subsample
        self.rnn_type = rnn_type
        self.bidir = bidir

        self.aux_output_layers = aux_output_layers

    def forward(
        self,
        rnn_input: torch.Tensor,
        rnn_len: torch.Tensor,
        prev_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """RNNP forward.

        Args:
            rnn_input: RNN input sequences. (B, T, D_in)
            rnn_len: RNN input sequences lengths. (B,)
            prev_states: RNN hidden states. [N x (B, T, D_proj)]

        Returns:
            rnn_output : RNN output sequences. (B, T, D_proj)
                         with or without intermediate RNN output sequences.
                         ((B, T, D_proj), [N x (B, T, D_proj)])
            rnn_len: RNN output sequences lengths. (B,)
            current_states: RNN hidden states. [N x (B, T, D_proj)]

        """
        aux_rnn_outputs = []
        aux_rnn_lens = []
        current_states = []

        for layer in range(self.elayers):
            if not isinstance(rnn_len, torch.Tensor):
                rnn_len = torch.tensor(rnn_len)

            pack_rnn_input = pack_padded_sequence(
                rnn_input, rnn_len.cpu(), batch_first=True
            )
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))

            if isinstance(rnn, (torch.nn.LSTM, torch.nn.GRU)):
                rnn.flatten_parameters()

            if prev_states is not None and rnn.bidirectional:
                prev_states = reset_backward_rnn_state(prev_states)

            pack_rnn_output, states = rnn(
                pack_rnn_input, hx=None if prev_states is None else prev_states[layer]
            )
            current_states.append(states)

            pad_rnn_output, rnn_len = pad_packed_sequence(
                pack_rnn_output, batch_first=True
            )

            sub = self.subsample[layer + 1]
            if sub > 1:
                pad_rnn_output = pad_rnn_output[:, ::sub]
                rnn_len = torch.tensor([int(i + 1) // sub for i in rnn_len])

            projection_layer = getattr(self, "bt%d" % layer)
            proj_rnn_output = projection_layer(
                pad_rnn_output.contiguous().view(-1, pad_rnn_output.size(2))
            )
            rnn_output = proj_rnn_output.view(
                pad_rnn_output.size(0), pad_rnn_output.size(1), -1
            )

            if layer in self.aux_output_layers:
                aux_rnn_outputs.append(rnn_output)
                aux_rnn_lens.append(rnn_len)

            if layer < self.elayers - 1:
                rnn_output = torch.tanh(self.dropout(rnn_output))

            rnn_input = rnn_output

        if aux_rnn_outputs:
            return (
                (rnn_output, aux_rnn_outputs),
                (rnn_len, aux_rnn_lens),
                current_states,
            )
        else:
            return rnn_output, rnn_len, current_states


class RNN(torch.nn.Module):
    """RNN module.

    Args:
        idim: Input dimension.
        rnn_type: RNN units type.
        elayers: Number of RNN layers.
        eunits: Number of units ((2 * eunits) if bidirectional)
        eprojs: Number of final projection units.
        dropout_rate: Dropout rate for RNN layers.
        aux_output_layers: List of layer IDs for auxiliary RNN output sequences.

    """

    def __init__(
        self,
        idim: int,
        rnn_type: str,
        elayers: int,
        eunits: int,
        eprojs: int,
        dropout_rate: float,
        aux_output_layers: List = [],
    ):
        """Initialize RNN module."""
        super().__init__()

        bidir = rnn_type[0] == "b"

        for i in range(elayers):
            if i == 0:
                input_dim = idim
            else:
                input_dim = eunits

            rnn_layer = torch.nn.LSTM if "lstm" in rnn_type else torch.nn.GRU
            rnn = rnn_layer(
                input_dim, eunits, num_layers=1, bidirectional=bidir, batch_first=True
            )

            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.elayers = elayers
        self.eunits = eunits
        self.eprojs = eprojs
        self.rnn_type = rnn_type
        self.bidir = bidir

        self.l_last = torch.nn.Linear(eunits, eprojs)

        self.aux_output_layers = aux_output_layers

    def forward(
        self,
        rnn_input: torch.Tensor,
        rnn_len: torch.Tensor,
        prev_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """RNN forward.

        Args:
            rnn_input: RNN input sequences. (B, T, D_in)
            rnn_len: RNN input sequences lengths. (B,)
            prev_states: RNN hidden states. [N x (B, T, D_proj)]

        Returns:
            rnn_output : RNN output sequences. (B, T, D_proj)
                         with or without intermediate RNN output sequences.
                         ((B, T, D_proj), [N x (B, T, D_proj)])
            rnn_len: RNN output sequences lengths. (B,)
            current_states: RNN hidden states. [N x (B, T, D_proj)]

        """
        aux_rnn_outputs = []
        aux_rnn_lens = []
        current_states = []

        for layer in range(self.elayers):
            if not isinstance(rnn_len, torch.Tensor):
                rnn_len = torch.tensor(rnn_len)

            pack_rnn_input = pack_padded_sequence(
                rnn_input, rnn_len.cpu(), batch_first=True
            )

            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))

            if isinstance(rnn, (torch.nn.LSTM, torch.nn.GRU)):
                rnn.flatten_parameters()

            if prev_states is not None and rnn.bidirectional:
                prev_states = reset_backward_rnn_state(prev_states)

            pack_rnn_output, states = rnn(
                pack_rnn_input, hx=None if prev_states is None else prev_states[layer]
            )
            current_states.append(states)

            rnn_output, rnn_len = pad_packed_sequence(pack_rnn_output, batch_first=True)

            if self.bidir:
                rnn_output = (
                    rnn_output[:, :, : self.eunits] + rnn_output[:, :, self.eunits :]
                )

            if layer in self.aux_output_layers:
                aux_proj_rnn_output = torch.tanh(
                    self.l_last(rnn_output.contiguous().view(-1, rnn_output.size(2)))
                )
                aux_rnn_output = aux_proj_rnn_output.view(
                    rnn_output.size(0), rnn_output.size(1), -1
                )

                aux_rnn_outputs.append(aux_rnn_output)
                aux_rnn_lens.append(rnn_len)

            if layer < self.elayers - 1:
                rnn_input = self.dropout(rnn_output)

        proj_rnn_output = torch.tanh(
            self.l_last(rnn_output.contiguous().view(-1, rnn_output.size(2)))
        )
        rnn_output = proj_rnn_output.view(rnn_output.size(0), rnn_output.size(1), -1)

        if aux_rnn_outputs:
            return (
                (rnn_output, aux_rnn_outputs),
                (rnn_len, aux_rnn_lens),
                current_states,
            )
        else:
            return rnn_output, rnn_len, current_states


def reset_backward_rnn_state(
    states: Union[torch.Tensor, List[Optional[torch.Tensor]]],
) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
    """Set backward BRNN states to zeroes.

    Args:
        states: Encoder hidden states.

    Returns:
        states: Encoder hidden states with backward set to zero.

    """
    if isinstance(states, list):
        for state in states:
            state[1::2] = 0.0
    else:
        states[1::2] = 0.0

    return states


class VGG2L(torch.nn.Module):
    """VGG-like module.

    Args:
        in_channel: number of input channels

    """

    def __init__(self, in_channel: int = 1):
        """Initialize VGG-like module."""
        super(VGG2L, self).__init__()

        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(
        self, feats: torch.Tensor, feats_len: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """VGG2L forward.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_len: Feature sequences lengths. (B, )

        Returns:
            vgg_out: VGG2L output sequences. (B, F // 4, 128 * D_feats // 4)
            vgg_out_len: VGG2L output sequences lengths. (B,)

        """
        feats = feats.view(
            feats.size(0),
            feats.size(1),
            self.in_channel,
            feats.size(2) // self.in_channel,
        ).transpose(1, 2)

        vgg1 = F.relu(self.conv1_1(feats))
        vgg1 = F.relu(self.conv1_2(vgg1))
        vgg1 = F.max_pool2d(vgg1, 2, stride=2, ceil_mode=True)

        vgg2 = F.relu(self.conv2_1(vgg1))
        vgg2 = F.relu(self.conv2_2(vgg2))
        vgg2 = F.max_pool2d(vgg2, 2, stride=2, ceil_mode=True)

        vgg_out = vgg2.transpose(1, 2)
        vgg_out = vgg_out.contiguous().view(
            vgg_out.size(0), vgg_out.size(1), vgg_out.size(2) * vgg_out.size(3)
        )

        if torch.is_tensor(feats_len):
            feats_len = feats_len.cpu().numpy()
        else:
            feats_len = np.array(feats_len, dtype=np.float32)

        vgg1_len = np.array(np.ceil(feats_len / 2), dtype=np.int64)
        vgg_out_len = np.array(
            np.ceil(np.array(vgg1_len, dtype=np.float32) / 2), dtype=np.int64
        ).tolist()

        return vgg_out, vgg_out_len, None


class Encoder(torch.nn.Module):
    """Encoder module.

    Args:
        idim: Input dimension.
        etype: Encoder units type.
        elayers: Number of encoder layers.
        eunits: Number of encoder units per layer.
        eprojs: Number of projection units per layer.
        subsample: Subsampling rate per layer.
        dropout_rate: Dropout rate for encoder layers.
        intermediate_encoder_layers: Layer IDs for auxiliary encoder output sequences.

    """

    def __init__(
        self,
        idim: int,
        etype: str,
        elayers: int,
        eunits: int,
        eprojs: int,
        subsample: np.ndarray,
        dropout_rate: float = 0.0,
        aux_enc_output_layers: List = [],
    ):
        """Initialize Encoder module."""
        super(Encoder, self).__init__()

        rnn_type = etype.lstrip("vgg").rstrip("p")
        in_channel = 1

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNNP(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            rnn_type,
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout_rate=dropout_rate,
                            aux_output_layers=aux_enc_output_layers,
                        ),
                    ]
                )
            else:
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNN(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            rnn_type,
                            elayers,
                            eunits,
                            eprojs,
                            dropout_rate=dropout_rate,
                            aux_output_layers=aux_enc_output_layers,
                        ),
                    ]
                )
            self.conv_subsampling_factor = 4
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        RNNP(
                            idim,
                            rnn_type,
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout_rate=dropout_rate,
                            aux_output_layers=aux_enc_output_layers,
                        )
                    ]
                )
            else:
                self.enc = torch.nn.ModuleList(
                    [
                        RNN(
                            idim,
                            rnn_type,
                            elayers,
                            eunits,
                            eprojs,
                            dropout_rate=dropout_rate,
                            aux_output_layers=aux_enc_output_layers,
                        )
                    ]
                )
            self.conv_subsampling_factor = 1

    def forward(
        self,
        feats: torch.Tensor,
        feats_len: torch.Tensor,
        prev_states: Optional[List[torch.Tensor]] = None,
    ):
        """Forward encoder.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_len: Feature sequences lengths. (B,)
            prev_states: Previous encoder hidden states. [N x (B, T, D_enc)]

        Returns:
            enc_out: Encoder output sequences. (B, T, D_enc)
                   with or without encoder intermediate output sequences.
                   ((B, T, D_enc), [N x (B, T, D_enc)])
            enc_out_len: Encoder output sequences lengths. (B,)
            current_states: Encoder hidden states. [N x (B, T, D_enc)]

        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        _enc_out = feats
        _enc_out_len = feats_len
        current_states = []
        for rnn_module, prev_state in zip(self.enc, prev_states):
            _enc_out, _enc_out_len, states = rnn_module(
                _enc_out,
                _enc_out_len,
                prev_states=prev_state,
            )
            current_states.append(states)

        if isinstance(_enc_out, tuple):
            enc_out, aux_enc_out = _enc_out[0], _enc_out[1]
            enc_out_len, aux_enc_out_len = _enc_out_len[0], _enc_out_len[1]

            enc_out_mask = to_device(enc_out, make_pad_mask(enc_out_len).unsqueeze(-1))
            enc_out = enc_out.masked_fill(enc_out_mask, 0.0)

            for i in range(len(aux_enc_out)):
                aux_mask = to_device(
                    aux_enc_out[i], make_pad_mask(aux_enc_out_len[i]).unsqueeze(-1)
                )
                aux_enc_out[i] = aux_enc_out[i].masked_fill(aux_mask, 0.0)

            return (
                (enc_out, aux_enc_out),
                (enc_out_len, aux_enc_out_len),
                current_states,
            )
        else:
            enc_out_mask = to_device(
                _enc_out, make_pad_mask(_enc_out_len).unsqueeze(-1)
            )

            return _enc_out.masked_fill(enc_out_mask, 0.0), _enc_out_len, current_states


def encoder_for(
    args: Namespace,
    idim: int,
    subsample: np.ndarray,
    aux_enc_output_layers: List = [],
) -> torch.nn.Module:
    """Instantiate a RNN encoder with specified arguments.

    Args:
        args: The model arguments.
        idim: Input dimension.
        subsample: Subsampling rate per layer.
        aux_enc_output_layers: Layer IDs for auxiliary encoder output sequences.

    Returns:
        : Encoder module.

    """
    return Encoder(
        idim,
        args.etype,
        args.elayers,
        args.eunits,
        args.eprojs,
        subsample,
        dropout_rate=args.dropout_rate,
        aux_enc_output_layers=aux_enc_output_layers,
    )
