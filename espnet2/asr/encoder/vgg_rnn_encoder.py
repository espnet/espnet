from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN, RNNP, VGG2L


class VGGRNNEncoder(AbsEncoder):
    """
        VGGRNNEncoder class for feature extraction in speech recognition tasks.

    This class combines a VGG (Visual Geometry Group) network with a Recurrent Neural Network (RNN)
    to process input features for Automatic Speech Recognition (ASR) tasks.

    Attributes:
        _output_size (int): The size of the output features.
        rnn_type (str): The type of RNN used ('lstm' or 'gru').
        bidirectional (bool): Whether the RNN is bidirectional.
        use_projection (bool): Whether to use projection layer after RNN.
        enc (torch.nn.ModuleList): List of encoder modules (VGG2L and RNN/RNNP).

    Args:
        input_size (int): The number of expected features in the input.
        rnn_type (str, optional): Type of RNN to use. Defaults to "lstm".
        bidirectional (bool, optional): If True, use bidirectional RNN. Defaults to True.
        use_projection (bool, optional): Whether to use projection layer. Defaults to True.
        num_layers (int, optional): Number of recurrent layers. Defaults to 4.
        hidden_size (int, optional): The number of features in the hidden state. Defaults to 320.
        output_size (int, optional): The number of output features. Defaults to 320.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        in_channel (int, optional): Number of input channels. Defaults to 1.

    Raises:
        ValueError: If rnn_type is not 'lstm' or 'gru'.

    Example:
        >>> input_size = 80
        >>> encoder = VGGRNNEncoder(input_size, rnn_type="lstm", num_layers=3)
        >>> input_tensor = torch.randn(32, 100, input_size)  # (batch, time, features)
        >>> input_lengths = torch.randint(50, 100, (32,))
        >>> output, output_lengths, _ = encoder(input_tensor, input_lengths)
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
        in_channel: int = 1,
    ):
        super().__init__()
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_projection = use_projection
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        # Subsample is not used for VGGRNN
        subsample = np.ones(num_layers + 1, dtype=np.int64)
        rnn_type = ("b" if bidirectional else "") + rnn_type
        if use_projection:
            self.enc = torch.nn.ModuleList(
                [
                    VGG2L(in_channel),
                    RNNP(
                        get_vgg2l_odim(input_size, in_channel=in_channel),
                        num_layers,
                        hidden_size,
                        output_size,
                        subsample,
                        dropout,
                        typ=rnn_type,
                    ),
                ]
            )

        else:
            self.enc = torch.nn.ModuleList(
                [
                    VGG2L(in_channel),
                    RNN(
                        get_vgg2l_odim(input_size, in_channel=in_channel),
                        num_layers,
                        hidden_size,
                        output_size,
                        dropout,
                        typ=rnn_type,
                    ),
                ]
            )

    def output_size(self) -> int:
        """
                Returns the output size of the encoder.

        Returns:
            int: The number of output features from the encoder.

        Example:
            >>> encoder = VGGRNNEncoder(input_size=80, output_size=320)
            >>> print(encoder.output_size())
            320
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
                Forward pass of the VGGRNNEncoder.

        This method processes the input through the VGG layers followed by RNN layers.

        Args:
            xs_pad (torch.Tensor): Padded input tensor of shape (batch, time, feat).
            ilens (torch.Tensor): Input lengths of each sequence in batch.
            prev_states (torch.Tensor, optional): Previous states for RNN layers. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - xs_pad (torch.Tensor): Output tensor after encoding.
                - ilens (torch.Tensor): Output lengths of each sequence in batch.
                - current_states (List[torch.Tensor]): List of current states for RNN layers.

        Example:
            >>> encoder = VGGRNNEncoder(input_size=80, output_size=320)
            >>> xs_pad = torch.randn(32, 100, 80)  # (batch, time, feat)
            >>> ilens = torch.randint(50, 100, (32,))
            >>> output, output_lengths, states = encoder(xs_pad, ilens)
            >>> print(output.shape)
            torch.Size([32, 25, 320])  # (batch, reduced_time, output_size)

        Note:
            The time dimension is typically reduced due to the VGG layers' stride operations.
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
