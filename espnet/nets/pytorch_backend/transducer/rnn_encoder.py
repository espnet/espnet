"""RNN encoder implementation for transducer-based models.

These classes are based on the ones in espnet.nets.pytorch_backend.rnn.encoders,
and modified to output intermediate layers representation based on a list of
layers given as input. These additional outputs are intended to be used with
auxiliary tasks.
It should be noted that, here, RNN class rely on a stack of 1-layer LSTM instead
of a multi-layer LSTM for that purpose.

"""

import argparse
import logging
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device


class RNNP(torch.nn.Module):
    """RNN with projection layer module.

    Args:
        idim: Dimension of inputs
        elayers: Dimension of encoder layers
        cdim: Number of units (results in cdim * 2 if bidirectional)
        hdim: Number of projection units
        subsample: List of subsampling number
        dropout: Dropout rate
        typ: RNN type
        aux_task_layer_list: List of layer ids for intermediate output

    """

    def __init__(
        self,
        idim: int,
        elayers: int,
        cdim: int,
        hdim: int,
        subsample: np.ndarray,
        dropout: float,
        typ: str = "blstm",
        aux_task_layer_list: List = [],
    ):
        """Initialize RNNP module."""
        super(RNNP, self).__init__()

        bidir = typ[0] == "b"
        for i in range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim

            RNN = torch.nn.LSTM if "lstm" in typ else torch.nn.GRU
            rnn = RNN(
                inputdim, cdim, num_layers=1, bidirectional=bidir, batch_first=True
            )

            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)

            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir
        self.dropout = dropout

        self.aux_task_layer_list = aux_task_layer_list

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, List], torch.Tensor]:
        """RNNP forward.

        Args:
            xs_pad: Batch of padded input sequences (B, Tmax, idim)
            ilens: Batch of lengths of input sequences (B)
            prev_state: Batch of previous RNN states

        Returns:
            : Batch of padded output sequences (B, Tmax, hdim)
                    or tuple w/ aux outputs ((B, Tmax, hdim), [L x (B, Tmax, hdim)])
            : Batch of lengths of output sequences (B)
            : Batch of hidden state sequences (B, Tmax, hdim)

        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))

        aux_xs_list = []
        elayer_states = []
        for layer in range(self.elayers):
            if not isinstance(ilens, torch.Tensor):
                ilens = torch.tensor(ilens)

            xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True)
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()

            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)

            ys, states = rnn(
                xs_pack, hx=None if prev_state is None else prev_state[layer]
            )
            elayer_states.append(states)

            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)

            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = torch.tensor([int(i + 1) // sub for i in ilens])

            projection_layer = getattr(self, "bt%d" % layer)
            projected = projection_layer(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)

            if layer in self.aux_task_layer_list:
                aux_xs_list.append(xs_pad)

            if layer < self.elayers - 1:
                xs_pad = torch.tanh(F.dropout(xs_pad, p=self.dropout))

        if aux_xs_list:
            return (xs_pad, aux_xs_list), ilens, elayer_states
        else:
            return xs_pad, ilens, elayer_states


class RNN(torch.nn.Module):
    """RNN module.

    Args:
        idim: Dimension of inputs
        elayers: Number of encoder layers
        cdim: Number of rnn units (resulted in cdim * 2 if bidirectional)
        hdim: Number of final projection units
        dropout: Dropout rate
        typ: The RNN type

    """

    def __init__(
        self,
        idim: int,
        elayers: int,
        cdim: int,
        hdim: int,
        dropout: float,
        typ: str = "blstm",
        aux_task_layer_list: List = [],
    ):
        """Initialize RNN module."""
        super(RNN, self).__init__()

        bidir = typ[0] == "b"

        for i in range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = cdim

            layer_type = torch.nn.LSTM if "lstm" in typ else torch.nn.GRU
            rnn = layer_type(
                inputdim, cdim, num_layers=1, bidirectional=bidir, batch_first=True
            )

            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)

        self.dropout = torch.nn.Dropout(p=dropout)

        self.elayers = elayers
        self.cdim = cdim
        self.hdim = hdim
        self.typ = typ
        self.bidir = bidir

        self.l_last = torch.nn.Linear(cdim, hdim)

        self.aux_task_layer_list = aux_task_layer_list

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, List], torch.Tensor]:
        """RNN forward.

        Args:
            xs_pad: Batch of padded input sequences (B, Tmax, idim)
            ilens: Batch of lengths of input sequences (B)
            prev_state: Batch of previous RNN states

        Returns:
            : Batch of padded output sequences (B, Tmax, hdim)
                    or tuple w/ aux outputs ((B, Tmax, hdim), [L x (B, Tmax, hdim)])
            : Batch of lengths of output sequences (B)
            : Batch of hidden state sequences (B, Tmax, hdim)

        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))

        aux_xs_list = []
        elayer_states = []
        for layer in range(self.elayers):
            if not isinstance(ilens, torch.Tensor):
                ilens = torch.tensor(ilens)

            xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True)

            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()

            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)

            xs, states = rnn(
                xs_pack, hx=None if prev_state is None else prev_state[layer]
            )
            elayer_states.append(states)

            xs_pad, ilens = pad_packed_sequence(xs, batch_first=True)

            if self.bidir:
                xs_pad = xs_pad[:, :, : self.cdim] + xs_pad[:, :, self.cdim :]

            if layer in self.aux_task_layer_list:
                aux_projected = torch.tanh(
                    self.l_last(xs_pad.contiguous().view(-1, xs_pad.size(2)))
                )
                aux_xs_pad = aux_projected.view(xs_pad.size(0), xs_pad.size(1), -1)

                aux_xs_list.append(aux_xs_pad)

            if layer < self.elayers - 1:
                xs_pad = self.dropout(xs_pad)

        projected = torch.tanh(
            self.l_last(xs_pad.contiguous().view(-1, xs_pad.size(2)))
        )
        xs_pad = projected.view(xs_pad.size(0), xs_pad.size(1), -1)

        if aux_xs_list:
            return (xs_pad, aux_xs_list), ilens, elayer_states
        else:
            return xs_pad, ilens, elayer_states


def reset_backward_rnn_state(
    states: Union[torch.Tensor, Tuple, List]
) -> Union[torch.Tensor, Tuple, List]:
    """Set backward BRNN states to zeroes.

    Args:
        states: RNN states

    Returns:
        states: RNN states with backward set to zeroes

    """
    if isinstance(states, (list, tuple)):
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

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor, **kwargs):
        """VGG2L forward.

        Args:
            xs_pad: Batch of padded input sequences (B, Tmax, D)
            ilens: Batch of lengths of input sequences (B)

        Returns:
            : Batch of padded output sequences (B, Tmax // 4, 128 * D // 4)
            : Batch of lengths of output sequences (B)

        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))

        xs_pad = xs_pad.view(
            xs_pad.size(0),
            xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel,
        ).transpose(1, 2)

        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64
        ).tolist()

        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )

        return xs_pad, ilens, None


class Encoder(torch.nn.Module):
    """Encoder module.

    Args:
        etype: Type of encoder network
        idim: Number of dimensions of encoder network
        elayers: Number of layers of encoder network
        eunits: Number of RNN units of encoder network
        eprojs: Number of projection units of encoder network
        subsample: List of subsampling numbers
        dropout: Dropout rate
        in_channel: Number of input channels

    """

    def __init__(
        self,
        etype: str,
        idim: int,
        elayers: int,
        eunits: int,
        eprojs: int,
        subsample: np.ndarray,
        dropout: float,
        in_channel: int = 1,
        aux_task_layer_list: List = [],
    ):
        """Initialize Encoder module."""
        super(Encoder, self).__init__()

        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ["lstm", "gru", "blstm", "bgru"]:
            logging.error("Error: need to specify an appropriate encoder architecture")

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNNP(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout,
                            typ=typ,
                            aux_task_layer_list=aux_task_layer_list,
                        ),
                    ]
                )
                logging.info("Use CNN-VGG + " + typ.upper() + "P for encoder")
            else:
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNN(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            dropout,
                            typ=typ,
                            aux_task_layer_list=aux_task_layer_list,
                        ),
                    ]
                )
                logging.info("Use CNN-VGG + " + typ.upper() + " for encoder")
            self.conv_subsampling_factor = 4
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        RNNP(
                            idim,
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout,
                            typ=typ,
                            aux_task_layer_list=aux_task_layer_list,
                        )
                    ]
                )
                logging.info(typ.upper() + " with every-layer projection for encoder")
            else:
                self.enc = torch.nn.ModuleList(
                    [
                        RNN(
                            idim,
                            elayers,
                            eunits,
                            eprojs,
                            dropout,
                            typ=typ,
                            aux_task_layer_list=aux_task_layer_list,
                        )
                    ]
                )
                logging.info(typ.upper() + " without projection for encoder")
            self.conv_subsampling_factor = 1

    def forward(self, xs_pad, ilens, prev_states=None):
        """Forward encoder.

        Args:
            xs_pad: Batch of padded input sequences (B, Tmax, idim)
            ilens: Batch of lengths of input sequences (B)
            prev_state: Batch of previous encoder hidden states (B, ??)

        Returns:
            : Batch of padded output sequences (B, Tmax, hdim)
                    or tuple w/ aux outputs ((B, Tmax, hdim), [L x (B, Tmax, hdim)])
            : Batch of lengths of output sequences (B)
            : Batch of hidden state sequences (B, Tmax, hdim)

        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(
                xs_pad,
                ilens,
                prev_state=prev_state,
            )
            current_states.append(states)

        if isinstance(xs_pad, tuple):
            final_xs_pad, aux_xs_list = xs_pad[0], xs_pad[1]

            mask = to_device(final_xs_pad, make_pad_mask(ilens).unsqueeze(-1))

            aux_xs_list = [layer.masked_fill(mask, 0.0) for layer in aux_xs_list]

            return (
                (
                    final_xs_pad.masked_fill(mask, 0.0),
                    aux_xs_list,
                ),
                ilens,
                current_states,
            )
        else:
            mask = to_device(xs_pad, make_pad_mask(ilens).unsqueeze(-1))

            return xs_pad.masked_fill(mask, 0.0), ilens, current_states


def encoder_for(
    args: argparse.Namespace,
    idim: Union[int, List],
    subsample: np.ndarray,
    aux_task_layer_list: List = [],
) -> Union[torch.nn.Module, List[torch.nn.Module]]:
    """Instantiate an encoder module given the program arguments.

    Args:
        args: The model arguments
        idim: Dimension of inputs or list of dimensions of inputs for each encoder
        subsample: subsample factors or list of subsample factors for each encoder

    Returns:
        : The encoder module or list of encoder modules

    """
    return Encoder(
        args.etype,
        idim,
        args.elayers,
        args.eunits,
        args.eprojs,
        subsample,
        args.dropout_rate,
        aux_task_layer_list=aux_task_layer_list,
    )
