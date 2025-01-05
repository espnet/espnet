from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.dprnn import DPRNN, merge_feature, split_feature
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class DPRNNSeparator(AbsSeparator):
    """
    DPRNNSeparator is a Dual-Path RNN (DPRNN) based separator for audio signals.

    This class implements a Dual-Path RNN architecture for separating audio signals
    from multiple speakers. It is designed to process complex input features and
    output the separated signals for each speaker, along with an optional noise
    estimate.

    Attributes:
        num_spk (int): Number of speakers.
        predict_noise (bool): Whether to output the estimated noise signal.
        segment_size (int): Dual-path segment size.
        num_outputs (int): Number of outputs, including noise if predicted.
        dprnn (DPRNN): Instance of the DPRNN model.

    Args:
        input_dim (int): Input feature dimension.
        rnn_type (str, optional): Type of RNN to use ('RNN', 'LSTM', 'GRU').
            Default is 'lstm'.
        bidirectional (bool, optional): Whether the inter-chunk RNN layers are
            bidirectional. Default is True.
        num_spk (int, optional): Number of speakers. Default is 2.
        predict_noise (bool, optional): Whether to output the estimated noise signal.
            Default is False.
        nonlinear (str, optional): Nonlinear function for mask estimation.
            Choose from 'relu', 'tanh', 'sigmoid'. Default is 'relu'.
        layer (int, optional): Number of stacked RNN layers. Default is 3.
        unit (int, optional): Dimension of the hidden state. Default is 512.
        segment_size (int, optional): Dual-path segment size. Default is 20.
        dropout (float, optional): Dropout ratio. Default is 0.0.

    Raises:
        ValueError: If the specified nonlinear function is not supported.

    Examples:
        >>> separator = DPRNNSeparator(input_dim=256, num_spk=2)
        >>> input_features = torch.randn(10, 100, 256)  # [Batch, Time, Features]
        >>> ilens = torch.tensor([100] * 10)  # Input lengths
        >>> masked, ilens, others = separator(input_features, ilens)

    Note:
        The `additional` argument in the forward method is not used in this model.

    Yields:
        masked (List[Union[torch.Tensor, ComplexTensor]]): List of separated signals
        for each speaker.
        ilens (torch.Tensor): Input lengths after processing.
        others (OrderedDict): Additional predicted data such as masks for each speaker.
    """

    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
    ):
        """Dual-Path RNN (DPRNN) Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            segment_size: dual-path segment size
            dropout: float, dropout ratio. Default is 0.
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.segment_size = segment_size

        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.dprnn = DPRNN(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            dropout=dropout,
            num_layers=layer,
            bidirectional=bidirectional,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
            Forward pass of the DPRNN Separator.

        This method processes the input features through the Dual-Path RNN (DPRNN)
        to estimate the masks for the specified number of speakers. It handles both
        real and complex input tensors.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): Encoded feature tensor of
                shape [B, T, N], where B is the batch size, T is the number of time
                frames, and N is the number of frequency bins.
            ilens (torch.Tensor): Input lengths tensor of shape [Batch], containing
                the lengths of each input sequence.
            additional (Optional[Dict]): Additional data included in the model.
                NOTE: This parameter is not used in this model.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor,
                   OrderedDict]: A tuple containing:
                - masked (List[Union[torch.Tensor, ComplexTensor]]): A list of
                  tensors of shape [(B, T, N), ...] where each tensor corresponds
                  to the input multiplied by the estimated mask for each speaker.
                - ilens (torch.Tensor): The input lengths tensor of shape (B,).
                - others (OrderedDict): A dictionary containing the predicted masks
                  for each speaker, e.g.:
                    - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    - ...,
                    - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

        Examples:
            >>> separator = DPRNNSeparator(input_dim=512, num_spk=2)
            >>> input_tensor = torch.randn(8, 100, 512)  # Batch of 8, 100 time frames
            >>> ilens = torch.tensor([100] * 8)  # All sequences are of length 100
            >>> masked, ilens_out, others = separator.forward(input_tensor, ilens)
            >>> print(len(masked))  # Should print 2 if num_spk=2
            >>> print(others.keys())  # Should include 'mask_spk1' and 'mask_spk2'

        Note:
            This implementation supports both real-valued and complex-valued input
            tensors. If the input tensor is complex, the magnitude is used for
            processing.

        Raises:
            ValueError: If an unsupported nonlinear activation function is provided.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, T
        segmented, rest = split_feature(
            feature, segment_size=self.segment_size
        )  # B, N, L, K

        processed = self.dprnn(segmented)  # B, N*num_spk, L, K

        processed = merge_feature(processed, rest)  # B, N*num_spk, T

        processed = processed.transpose(1, 2)  # B, T, N*num_spk
        processed = processed.view(B, T, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)

        if self.predict_noise:
            *masks, mask_noise = masks
        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
