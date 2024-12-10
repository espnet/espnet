from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class ConformerSeparator(AbsSeparator):
    """
        ConformerSeparator is a neural network module that performs source separation
    using the Conformer architecture. It processes audio features and estimates
    masks for multiple speakers, optionally predicting noise as well.

    Attributes:
        num_spk (int): The number of speakers to separate.
        predict_noise (bool): Flag indicating whether to predict the noise signal.

    Args:
        input_dim (int): Input feature dimension.
        num_spk (int, optional): Number of speakers. Defaults to 2.
        predict_noise (bool, optional): Whether to output the estimated noise signal.
            Defaults to False.
        adim (int, optional): Dimension of attention. Defaults to 384.
        aheads (int, optional): The number of heads in multi-head attention. Defaults to 4.
        layers (int, optional): The number of transformer blocks. Defaults to 6.
        linear_units (int, optional): The number of units in position-wise feed forward.
            Defaults to 1536.
        positionwise_layer_type (str, optional): Type of position-wise layer. Defaults to
            "linear".
        positionwise_conv_kernel_size (int, optional): Kernel size of position-wise
            convolutional layer. Defaults to 1.
        normalize_before (bool, optional): Whether to use layer normalization before the
            first block. Defaults to False.
        concat_after (bool, optional): Whether to concatenate attention layer's input
            and output. Defaults to False.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        input_layer (Union[str, torch.nn.Module], optional): Input layer type. Defaults to
            "linear".
        positional_dropout_rate (float, optional): Dropout rate after adding positional
            encoding. Defaults to 0.1.
        attention_dropout_rate (float, optional): Dropout rate in attention. Defaults to 0.1.
        nonlinear (str, optional): Nonlinear function for mask estimation. Options are
            'relu', 'tanh', 'sigmoid'. Defaults to 'relu'.
        conformer_pos_enc_layer_type (str, optional): Encoder positional encoding layer
            type. Defaults to "rel_pos".
        conformer_self_attn_layer_type (str, optional): Encoder attention layer type.
            Defaults to "rel_selfattn".
        conformer_activation_type (str, optional): Encoder activation function type.
            Defaults to "swish".
        use_macaron_style_in_conformer (bool, optional): Whether to use macaron style
            for position-wise layer. Defaults to True.
        use_cnn_in_conformer (bool, optional): Whether to use convolution module.
            Defaults to True.
        conformer_enc_kernel_size (int, optional): Kernel size of convolution module.
            Defaults to 7.
        padding_idx (int, optional): Padding index for input_layer=embed. Defaults to -1.

    Raises:
        ValueError: If the specified nonlinear function is not supported.

    Examples:
        >>> separator = ConformerSeparator(input_dim=128, num_spk=2)
        >>> input_features = torch.randn(10, 50, 128)  # (Batch, Time, Features)
        >>> ilens = torch.tensor([50] * 10)  # Input lengths for each sample
        >>> masked, ilens_out, masks = separator(input_features, ilens)
        >>> print(len(masks))  # Should be 2 if num_spk is 2

    Note:
        This module assumes that the input features are either real-valued tensors or
        complex tensors.
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
        input_layer: str = "linear",
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        nonlinear: str = "relu",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_enc_kernel_size: int = 7,
        padding_idx: int = -1,
    ):
        """Conformer separator.

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            adim (int): Dimension of attention.
            aheads (int): The number of heads of multi head attention.
            linear_units (int): The number of units of position-wise feed forward.
            layers (int): The number of transformer blocks.
            dropout_rate (float): Dropout rate.
            input_layer (Union[str, torch.nn.Module]): Input layer type.
            attention_dropout_rate (float): Dropout rate in attention.
            positional_dropout_rate (float): Dropout rate after adding
                                             positional encoding.
            normalize_before (bool): Whether to use layer_norm before the first block.
            concat_after (bool): Whether to concat attention layer's input and output.
                if True, additional linear will be applied.
                i.e. x -> x + linear(concat(x, att(x)))
                if False, no additional linear will be applied. i.e. x -> x + att(x)
            conformer_pos_enc_layer_type(str): Encoder positional encoding layer type.
            conformer_self_attn_layer_type (str): Encoder attention layer type.
            conformer_activation_type(str): Encoder activation function type.
            positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
            positionwise_conv_kernel_size (int): Kernel size of
                                                 positionwise conv1d layer.
            use_macaron_style_in_conformer (bool): Whether to use macaron style for
                                                   positionwise layer.
            use_cnn_in_conformer (bool): Whether to use convolution module.
            conformer_enc_kernel_size(int): Kernerl size of convolution module.
            padding_idx (int): Padding idx for input_layer=embed.
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.conformer = ConformerEncoder(
            idim=input_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            pos_enc_layer_type=conformer_pos_enc_layer_type,
            selfattention_layer_type=conformer_self_attn_layer_type,
            activation_type=conformer_activation_type,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_enc_kernel_size,
            padding_idx=padding_idx,
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
        Performs the forward pass of the Conformer separator.

        This method takes encoded feature inputs and processes them through
        the Conformer architecture to produce separated audio masks for each
        speaker, along with the estimated noise if specified.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): Encoded feature tensor
                of shape [B, T, N], where B is the batch size, T is the
                sequence length, and N is the number of features.
            ilens (torch.Tensor): A tensor containing the lengths of the input
                sequences [Batch].
            additional (Optional[Dict]): A dictionary for additional data that
                may be included in the model. NOTE: This parameter is not used
                in this implementation.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor,
            OrderedDict]: A tuple containing:
                - masked (List[Union[torch.Tensor, ComplexTensor]]): A list of
                  tensors representing the separated audio signals for each
                  speaker of shape [(B, T, N), ...].
                - ilens (torch.Tensor): A tensor containing the lengths of the
                  output sequences [Batch].
                - others (OrderedDict): A dictionary of predicted data, e.g.,
                  masks for each speaker:
                    - 'mask_spk1': torch.Tensor(Batch, Frames, Freq)
                    - 'mask_spk2': torch.Tensor(Batch, Frames, Freq)
                    ...
                    - 'mask_spkn': torch.Tensor(Batch, Frames, Freq)
                    If noise prediction is enabled, an additional entry:
                    - 'noise1': torch.Tensor(Batch, Frames, Freq)

        Examples:
            >>> separator = ConformerSeparator(input_dim=64, num_spk=2)
            >>> input_tensor = torch.rand(8, 100, 64)  # Example input
            >>> input_lengths = torch.tensor([100] * 8)  # All sequences are 100
            >>> masks, lengths, others = separator.forward(input_tensor, input_lengths)

        Note:
            Ensure that the input tensor is either a real-valued tensor or a
            ComplexTensor. The function will handle both types appropriately.

        Raises:
            ValueError: If the input tensor is not of type torch.Tensor or
            ComplexTensor.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        # prepare pad_mask for transformer
        pad_mask = make_non_pad_mask(ilens).unsqueeze(1).to(feature.device)

        x, ilens = self.conformer(feature, pad_mask)

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
