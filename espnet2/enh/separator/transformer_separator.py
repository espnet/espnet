from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
)

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class TransformerSeparator(AbsSeparator):
    """
        TransformerSeparator is a class that implements a Transformer-based speech
    separator. It inherits from the AbsSeparator class and is designed to separate
    audio signals from multiple speakers.

    Attributes:
        num_spk (int): Number of speakers in the input audio.
        predict_noise (bool): Indicates whether to predict the noise signal.

    Args:
        input_dim (int): Input feature dimension.
        num_spk (int, optional): Number of speakers (default is 2).
        predict_noise (bool, optional): Whether to output the estimated noise
            signal (default is False).
        adim (int, optional): Dimension of attention (default is 384).
        aheads (int, optional): Number of heads for multi-head attention
            (default is 4).
        layers (int, optional): Number of transformer blocks (default is 6).
        linear_units (int, optional): Number of units in position-wise feed
            forward (default is 1536).
        positionwise_layer_type (str, optional): Type of position-wise layer
            ("linear", "conv1d", or "conv1d-linear", default is "linear").
        positionwise_conv_kernel_size (int, optional): Kernel size of
            position-wise conv1d layer (default is 1).
        normalize_before (bool, optional): Whether to use layer normalization
            before the first block (default is False).
        concat_after (bool, optional): Whether to concatenate the input and
            output of the attention layer (default is False).
        dropout_rate (float, optional): Dropout rate (default is 0.1).
        positional_dropout_rate (float, optional): Dropout rate after adding
            positional encoding (default is 0.1).
        attention_dropout_rate (float, optional): Dropout rate in attention
            (default is 0.1).
        use_scaled_pos_enc (bool, optional): Use scaled positional encoding or
            not (default is True).
        nonlinear (str, optional): Nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid' (default is 'relu').

    Raises:
        ValueError: If the nonlinear function is not one of 'sigmoid', 'relu',
            or 'tanh'.

    Examples:
        # Create a TransformerSeparator instance
        separator = TransformerSeparator(input_dim=256, num_spk=2)

        # Forward pass with a sample input
        input_tensor = torch.randn(10, 100, 256)  # Batch of 10, 100 time steps
        input_lengths = torch.tensor([100] * 10)  # All sequences are of length 100
        masked_outputs, lengths, others = separator(input_tensor, input_lengths)

    Note:
        This class requires PyTorch and the espnet2 library for its operation.
    """

    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        predict_noise: bool = False,
        adim: int = 384,
        aheads: int = 4,
        layers: int = 6,
        linear_units: int = 1536,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        normalize_before: bool = False,
        concat_after: bool = False,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        use_scaled_pos_enc: bool = True,
        nonlinear: str = "relu",
    ):
        """Transformer separator.

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            adim (int): Dimension of attention.
            aheads (int): The number of heads of multi head attention.
            linear_units (int): The number of units of position-wise feed forward.
            layers (int): The number of transformer blocks.
            dropout_rate (float): Dropout rate.
            attention_dropout_rate (float): Dropout rate in attention.
            positional_dropout_rate (float): Dropout rate after adding
                                             positional encoding.
            normalize_before (bool): Whether to use layer_norm before the first block.
            concat_after (bool): Whether to concat attention layer's input and output.
                if True, additional linear will be applied.
                i.e. x -> x + linear(concat(x, att(x)))
                if False, no additional linear will be applied. i.e. x -> x + att(x)
            positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
            positionwise_conv_kernel_size (int): Kernel size of
                                                 positionwise conv1d layer.
            use_scaled_pos_enc (bool) : use scaled positional encoding or not
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        pos_enc_class = (
            ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding
        )
        self.transformer = TransformerEncoder(
            idim=input_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers,
            input_layer="linear",
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(adim, input_dim) for _ in range(num_outputs)]
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
        Perform the forward pass of the Transformer separator.

        This method takes the encoded features and processes them through the
        Transformer model to produce speaker masks. It can handle both real
        and complex input tensors.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): Encoded feature of shape
                [B, T, N], where B is the batch size, T is the number of time
                frames, and N is the number of frequency bins.
            ilens (torch.Tensor): A tensor of input lengths of shape [Batch],
                indicating the length of each input sequence.
            additional (Optional[Dict]): A dictionary containing other data
                included in the model. Note: This is not used in this model.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor,
            OrderedDict]: A tuple containing:
                - masked (List[Union[torch.Tensor, ComplexTensor]]): A list of
                  tensors, each of shape (B, T, N), representing the input
                  features masked by the estimated speaker masks.
                - ilens (torch.Tensor): A tensor of shape (B,) containing the
                  lengths of the output sequences.
                - others (OrderedDict): A dictionary of predicted data,
                  including masks for each speaker. The keys are:
                    - 'mask_spk1': torch.Tensor of shape (Batch, Frames, Freq)
                    - 'mask_spk2': torch.Tensor of shape (Batch, Frames, Freq)
                    - ...
                    - 'mask_spkn': torch.Tensor of shape (Batch, Frames, Freq)
                    - 'noise1': torch.Tensor of shape (Batch, Frames, Freq)
                      (if predict_noise is True).

        Examples:
            >>> separator = TransformerSeparator(input_dim=256, num_spk=2)
            >>> input_tensor = torch.randn(8, 100, 256)  # Batch of 8
            >>> ilens = torch.tensor([100] * 8)  # All sequences are of length 100
            >>> masked, lengths, masks = separator.forward(input_tensor, ilens)

        Note:
            The input tensor can be either a real-valued tensor or a complex
            tensor. If a complex tensor is provided, the absolute values will
            be computed before processing.

        Raises:
            ValueError: If the nonlinear activation function is not one of
            'sigmoid', 'relu', or 'tanh'.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        # prepare pad_mask for transformer
        pad_mask = make_non_pad_mask(ilens).unsqueeze(1).to(feature.device)

        x, ilens = self.transformer(feature, pad_mask)

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
