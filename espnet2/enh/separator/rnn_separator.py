from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class RNNSeparator(AbsSeparator):
    """
    RNN-based separator for audio source separation.

    This class implements a Recurrent Neural Network (RNN) based separator
    for separating audio signals from multiple speakers. The model can
    estimate noise signals in addition to speaker masks.

    Attributes:
        num_spk (int): Number of speakers.
        predict_noise (bool): Whether to output the estimated noise signal.

    Args:
        input_dim (int): Input feature dimension.
        rnn_type (str): Type of RNN to use. Options include 'blstm', 'lstm', etc.
        num_spk (int): Number of speakers. Default is 2.
        predict_noise (bool): If True, outputs estimated noise signal. Default is False.
        nonlinear (str): Nonlinear function for mask estimation. Options are
                         'relu', 'tanh', 'sigmoid'. Default is 'sigmoid'.
        layer (int): Number of stacked RNN layers. Default is 3.
        unit (int): Dimension of the hidden state. Default is 512.
        dropout (float): Dropout ratio. Default is 0.

    Raises:
        ValueError: If an unsupported nonlinear function is specified.

    Examples:
        # Initialize the RNN separator
        separator = RNNSeparator(input_dim=512, num_spk=2, predict_noise=True)

        # Forward pass
        input_tensor = torch.randn(10, 100, 512)  # Example input [B, T, N]
        ilens = torch.tensor([100] * 10)  # Example input lengths
        masked, ilens_out, others = separator(input_tensor, ilens)

        # Accessing the masks for each speaker
        mask_spk1 = others['mask_spk1']
        mask_spk2 = others['mask_spk2']

        # Forward streaming
        streaming_output, states, others_stream = separator.forward_streaming(
            input_frame=torch.randn(10, 1, 512)
        )
    """

    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "sigmoid",
        layer: int = 3,
        unit: int = 512,
        dropout: float = 0.0,
    ):
        """RNN Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            dropout: float, dropout ratio. Default is 0.
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.rnn = RNN(
            idim=input_dim,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(unit, input_dim) for _ in range(num_outputs)]
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
        Perform the forward pass of the RNN Separator.

        This method processes the input features through the RNN and applies
        the estimated masks to the input to separate the speakers.

        Args:
            input (Union[torch.Tensor, ComplexTensor]):
                Encoded feature tensor of shape [B, T, N], where B is the
                batch size, T is the number of time frames, and N is the
                number of features.
            ilens (torch.Tensor):
                A tensor containing the lengths of the input sequences
                for each batch, shape [Batch].
            additional (Optional[Dict]):
                A dictionary containing other data included in the model.
                NOTE: This argument is not used in this model.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor,
                  OrderedDict]:
                A tuple containing:
                - masked (List[Union[torch.Tensor, ComplexTensor]]):
                    A list of tensors where each tensor has shape
                    (B, T, N) corresponding to the separated signals for
                    each speaker.
                - ilens (torch.Tensor):
                    A tensor containing the lengths of the output sequences,
                    shape (B,).
                - others (OrderedDict):
                    A dictionary containing predicted data such as masks:
                    - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                    - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                    ...
                    - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

        Examples:
            >>> rnn_separator = RNNSeparator(input_dim=64, num_spk=2)
            >>> input_tensor = torch.randn(10, 50, 64)  # Batch of 10
            >>> ilens = torch.tensor([50] * 10)  # All sequences have length 50
            >>> masked, lengths, masks = rnn_separator.forward(input_tensor, ilens)
            >>> print(len(masked))  # Should be 2 (for 2 speakers)
            >>> print(masks.keys())  # Should show keys for masks of each speaker

        Note:
            The input can be either a real-valued tensor or a complex tensor.
            If the input is complex, the absolute value will be used as the
            feature for the RNN.

        Raises:
            ValueError: If the input is not a torch.Tensor or ComplexTensor.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        x, ilens, _ = self.rnn(feature, ilens)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

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

    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        """
        Perform the forward pass for streaming input.

        This method processes a single frame of input data, allowing for
        streaming capabilities in the RNN separator. It computes the output
        masks for each speaker, as well as any predicted noise.

        Args:
            input_frame (torch.Tensor): The input frame with shape [B, 1, N],
                where B is the batch size and N is the number of features.
            states (Optional): The hidden states of the RNN, used for
                maintaining context across frames. If None, initializes new
                states.

        Returns:
            masked (List[Union[torch.Tensor, ComplexTensor]]): List of tensors
                where each tensor has shape [B, 1, N] representing the
                separated signals for each speaker.
            states: Updated hidden states of the RNN for the next frame.
            others (OrderedDict): Contains predicted data, such as masks:
                OrderedDict[
                    'mask_spk1': torch.Tensor(Batch, 1, Freq),
                    'mask_spk2': torch.Tensor(Batch, 1, Freq),
                    ...
                    'mask_spkn': torch.Tensor(Batch, 1, Freq),
                ]
                If predict_noise is True, it will also include:
                'noise1': torch.Tensor(Batch, 1, N)
                which represents the estimated noise signal.

        Examples:
            >>> separator = RNNSeparator(input_dim=128, num_spk=2)
            >>> input_frame = torch.randn(4, 1, 128)  # Batch of 4
            >>> masks, states, others = separator.forward_streaming(input_frame)

        Note:
            This method is designed for real-time processing of audio streams
            and is optimized for low-latency applications.
        """
        # input_frame # B, 1, N

        # if complex spectrum,
        if is_complex(input_frame):
            feature = abs(input_frame)
        else:
            feature = input_frame

        ilens = torch.ones(feature.shape[0], device=feature.device)

        x, _, states = self.rnn(feature, ilens, states)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input_frame * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, states, others
