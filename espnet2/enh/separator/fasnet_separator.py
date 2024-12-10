from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from packaging.version import parse as V

from espnet2.enh.layers.fasnet import FaSNet_TAC
from espnet2.enh.layers.ifasnet import iFaSNet
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class FaSNetSeparator(AbsSeparator):
    """
        FaSNetSeparator is a Filter-and-sum Network (FaSNet) Separator that inherits
    from the AbsSeparator class. This model is designed for separating audio signals
    based on the specified number of speakers and supports both the original FaSNet
    and the Implicit FaSNet architectures.

    Attributes:
        num_spk (int): The number of speakers.

    Args:
        input_dim (int): Required by AbsSeparator. Not used in this model.
        enc_dim (int): Encoder dimension.
        feature_dim (int): Feature dimension.
        hidden_dim (int): Hidden dimension in DPRNN.
        layer (int): Number of DPRNN blocks in iFaSNet.
        segment_size (int): Dual-path segment size.
        num_spk (int): Number of speakers.
        win_len (int): Window length in milliseconds.
        context_len (int): Context length in milliseconds.
        fasnet_type (str): 'fasnet' or 'ifasnet'. Select from origin fasnet or
            Implicit fasnet.
        dropout (float, optional): Dropout rate. Default is 0.0.
        sr (int, optional): Sample rate of input audio. Default is 16000.
        predict_noise (bool, optional): Whether to output the estimated noise signal.
            Default is False.

    Methods:
        forward(input: torch.Tensor, ilens: torch.Tensor,
                additional: Optional[Dict] = None) -> Tuple[List[torch.Tensor],
                torch.Tensor, OrderedDict]:
            Performs the forward pass of the model.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]: A tuple containing
        the separated audio signals, input lengths, and other predicted data
        (e.g., masks).

    Raises:
        AssertionError: If the input tensor does not have the expected shape
        (Batch, samples, channels).

    Examples:
        # Initialize the separator
        separator = FaSNetSeparator(
            input_dim=1,
            enc_dim=256,
            feature_dim=256,
            hidden_dim=512,
            layer=6,
            segment_size=10,
            num_spk=2,
            win_len=20,
            context_len=10,
            fasnet_type='fasnet',
            dropout=0.1,
            sr=16000,
            predict_noise=True
        )

        # Forward pass
        input_tensor = torch.randn(4, 16000, 1)  # (Batch, samples, channels)
        ilens = torch.tensor([16000, 16000, 16000, 16000])  # Input lengths
        separated, lengths, others = separator.forward(input_tensor, ilens)
    """

    def __init__(
        self,
        input_dim: int,
        enc_dim: int,
        feature_dim: int,
        hidden_dim: int,
        layer: int,
        segment_size: int,
        num_spk: int,
        win_len: int,
        context_len: int,
        fasnet_type: str,
        dropout: float = 0.0,
        sr: int = 16000,
        predict_noise: bool = False,
    ):
        """Filter-and-sum Network (FaSNet) Separator

        Args:
            input_dim: required by AbsSeparator. Not used in this model.
            enc_dim: encoder dimension
            feature_dim: feature dimension
            hidden_dim: hidden dimension in DPRNN
            layer: number of DPRNN blocks in iFaSNet
            segment_size: dual-path segment size
            num_spk: number of speakers
            win_len: window length in millisecond
            context_len: context length in millisecond
            fasnet_type: 'fasnet' or 'ifasnet'.
                Select from origin fasnet or Implicit fasnet
            dropout: dropout rate. Default is 0.
            sr: samplerate of input audio
            predict_noise: whether to output the estimated noise signal
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        assert fasnet_type in ["fasnet", "ifasnet"], "only support fasnet and ifasnet"

        FASNET = FaSNet_TAC if fasnet_type == "fasnet" else iFaSNet

        self.fasnet = FASNET(
            enc_dim=enc_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            layer=layer,
            segment_size=segment_size,
            nspk=num_spk + 1 if predict_noise else num_spk,
            win_len=win_len,
            context_len=context_len,
            sr=sr,
            dropout=dropout,
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """
        Perform the forward pass of the FaSNet separator.

        This method processes the input audio tensor to separate the sources
        based on the model architecture defined during initialization. It
        takes in the input audio signal and its corresponding lengths, and
        returns the separated sources along with their lengths and additional
        predicted data such as masks.

        Args:
            input (torch.Tensor): A tensor of shape (Batch, samples, channels)
                representing the input audio signals.
            ilens (torch.Tensor): A tensor of shape (Batch,) containing the
                lengths of each input signal in the batch.
            additional (Dict or None): A dictionary for any additional data
                that may be included in the model. Note that this parameter is
                not used in this implementation.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]: A tuple
            containing:
                - separated (List[torch.Tensor]): A list of tensors where each
                  tensor represents the separated audio for each speaker.
                  Shape: [(B, T, N), ...]
                - ilens (torch.Tensor): A tensor of shape (B,) containing
                  the lengths of the separated signals.
                - others (OrderedDict): A dictionary containing predicted data,
                  e.g., masks for each speaker:
                    - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    - ...
                    - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

        Raises:
            AssertionError: If the input tensor does not have 3 dimensions.

        Examples:
            >>> separator = FaSNetSeparator(input_dim=1, enc_dim=128,
            ...                             feature_dim=256, hidden_dim=512,
            ...                             layer=6, segment_size=400,
            ...                             num_spk=2, win_len=25,
            ...                             context_len=50, fasnet_type='fasnet')
            >>> input_tensor = torch.randn(8, 16000, 1)  # Batch of 8 audio signals
            >>> input_lengths = torch.tensor([16000]*8)  # Lengths of each signal
            >>> separated_sources, lengths, masks = separator.forward(input_tensor,
            ...                                                       input_lengths)

        Note:
            Ensure that the input tensor has the correct shape and that the
            number of speakers is set appropriately during initialization.
        """

        assert input.dim() == 3, "only support input shape: (Batch, samples, channels)"
        # currently only support for fixed-array

        input = input.permute(0, 2, 1)

        none_mic = torch.zeros(1, dtype=input.dtype)

        separated = self.fasnet(input, none_mic)

        separated = list(separated.unbind(dim=1))

        others = {}
        if self.predict_noise:
            *separated, noise = separated
            others["noise1"] = noise

        return separated, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
