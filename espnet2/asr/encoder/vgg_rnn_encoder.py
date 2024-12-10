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
    VGGRNNEncoder class for sequence-to-sequence modeling using VGG and RNN.

    This encoder combines VGG-based feature extraction with a recurrent neural
    network (RNN) architecture. It is designed to process sequences of
    input features and produce a sequence of output features, which can be
    used for various tasks such as automatic speech recognition (ASR).

    Attributes:
        output_size (int): The number of output features from the encoder.
        rnn_type (str): Type of RNN used, can be 'lstm' or 'gru'.
        bidirectional (bool): If True, the RNN will be bidirectional.
        use_projection (bool): If True, a projection layer will be used.

    Args:
        input_size (int): The number of expected features in the input.
        rnn_type (str, optional): The type of RNN to use ('lstm' or 'gru').
            Defaults to 'lstm'.
        bidirectional (bool, optional): If True, the RNN will be bidirectional.
            Defaults to True.
        use_projection (bool, optional): Whether to use a projection layer.
            Defaults to True.
        num_layers (int, optional): Number of recurrent layers. Defaults to 4.
        hidden_size (int, optional): The number of hidden features. Defaults to 320.
        output_size (int, optional): The number of output features. Defaults to 320.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        in_channel (int, optional): Number of input channels. Defaults to 1.

    Raises:
        ValueError: If an unsupported RNN type is specified.

    Examples:
        encoder = VGGRNNEncoder(input_size=80)
        xs_pad = torch.randn(10, 32, 80)  # (sequence_length, batch_size, features)
        ilens = torch.tensor([32] * 10)  # input lengths
        output, lengths, states = encoder(xs_pad, ilens)

    Note:
        The input features should be padded and properly masked before passing
        them to the forward method.

    Todo:
        - Add support for additional RNN types if needed.
        - Implement additional error handling for input dimensions.
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
        Get the output size of the encoder.

        This method returns the number of output features produced by the
        encoder, which is set during the initialization of the VGGRNNEncoder
        class. The output size is crucial for determining the dimensionality
        of the data passed to subsequent layers in a neural network.

        Returns:
            int: The number of output features.

        Examples:
            encoder = VGGRNNEncoder(input_size=128, output_size=256)
            output_size = encoder.output_size()
            print(output_size)  # Output: 256

        Note:
            The output size is defined during the instantiation of the
            VGGRNNEncoder class and can be used to ensure compatibility
            with subsequent layers.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes input tensors through the VGGRNNEncoder.

        This method takes padded input sequences and their corresponding lengths,
        along with previous states, and passes them through the encoder
        modules. It returns the processed output, updated input lengths,
        and the current states of the RNN.

        Args:
            xs_pad (torch.Tensor): Padded input tensor of shape (T, N, D),
                where T is the sequence length, N is the batch size, and D
                is the input feature dimension.
            ilens (torch.Tensor): Tensor containing the actual lengths of
                each sequence in the batch of shape (N,).
            prev_states (torch.Tensor, optional): Previous states of the
                RNN, default is None. If None, initializes states to None
                for each encoder module.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The output tensor after processing (shape: (T, N, D)).
                - The updated input lengths tensor (shape: (N,)).
                - A list of current states for each encoder module.

        Examples:
            >>> encoder = VGGRNNEncoder(input_size=40)
            >>> xs_pad = torch.randn(100, 32, 40)  # (T, N, D)
            >>> ilens = torch.tensor([100] * 32)    # (N,)
            >>> output, updated_ilens, states = encoder.forward(xs_pad, ilens)

        Note:
            The output tensor will have the same number of features as
            specified in the output_size attribute of the encoder.

        Raises:
            AssertionError: If the length of prev_states does not match
            the number of encoder modules.
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
