from typing import Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.diar.layers.tcn_nomask import TemporalConvNet
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class TCNSeparatorNomask(AbsSeparator):
    """
    TCNSeparatorNomask is a temporal convolutional network separator that processes
    audio features for speaker diarization tasks without estimating masks. It is
    designed to output intermediate bottleneck features that can be used as input
    to the diarization branch.

    This separator is equivalent to the TCNSeparator but omits the mask estimation
    component. The output features can be utilized by subsequent modules, such as
    the MultiMask module, for further processing.

    Attributes:
        output_dim (int): The dimensionality of the output bottleneck features.

    Args:
        input_dim (int): Input feature dimension.
        layer (int, optional): Number of layers in each stack. Default is 8.
        stack (int, optional): Number of stacks. Default is 3.
        bottleneck_dim (int, optional): Bottleneck dimension. Default is 128.
        hidden_dim (int, optional): Number of convolution channels. Default is 512.
        kernel (int, optional): Kernel size. Default is 3.
        causal (bool, optional): Whether to use causal convolutions. Default is False.
        norm_type (str, optional): Normalization type, choose from 'BN', 'gLN', 'cLN'.

    Examples:
        >>> separator = TCNSeparatorNomask(input_dim=64, layer=8, stack=3)
        >>> input_tensor = torch.randn(10, 100, 64)  # [Batch, Time, Features]
        >>> ilens = torch.tensor([100] * 10)  # Input lengths
        >>> feats, lengths = separator(input_tensor, ilens)
        >>> print(feats.shape)  # Output shape should be [10, 100, bottleneck_dim]

    Note:
        Ensure that the input tensor is in the shape of [Batch, Time, Features].
        This class requires PyTorch version 1.9.0 or higher for proper functionality.

    Raises:
        ValueError: If input dimensions do not match the expected shape.
    """
    def __init__(
        self,
        input_dim: int,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
    ):
        """Temporal Convolution Separator

        Note that this separator is equivalent to TCNSeparator except
        for not having the mask estimation part.
        This separator outputs the intermediate bottleneck feats
        (which is used as the input to diarization branch in enh_diar task).
        This separator is followed by MultiMask module,
        which estimates the masks.

        Args:
            input_dim: input feature dimension
            layer: int, number of layers in each stack.
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
        """
        super().__init__()

        self.tcn = TemporalConvNet(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            norm_type=norm_type,
            causal=causal,
        )

        self._output_dim = bottleneck_dim

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method for the TCNSeparatorNomask class.

        This method processes the input features through the temporal convolutional 
        network (TCN) and returns the extracted bottleneck features along with 
        the input lengths. It can handle both standard tensors and complex tensors 
        by taking the absolute value of the complex input.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): Encoded feature tensor 
                of shape [B, T, N], where B is the batch size, T is the sequence 
                length, and N is the feature dimension.
            ilens (torch.Tensor): A tensor of input lengths of shape [Batch] 
                representing the lengths of each input sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feats (torch.Tensor): Extracted bottleneck features of shape 
                [B, T, bottleneck_dim].
                - ilens (torch.Tensor): The input lengths tensor of shape [Batch].

        Examples:
            >>> separator = TCNSeparatorNomask(input_dim=64)
            >>> input_tensor = torch.randn(10, 50, 64)  # Batch of 10, seq len 50
            >>> input_lengths = torch.tensor([50] * 10)  # All sequences have length 50
            >>> feats, lengths = separator.forward(input_tensor, input_lengths)
            >>> print(feats.shape)  # Should print: torch.Size([10, 50, bottleneck_dim])
            >>> print(lengths.shape)  # Should print: torch.Size([10])

        Note:
            The method first checks if the input is a complex tensor. If it is, 
            the absolute value is computed before processing. The features are 
            transposed to match the expected input shape for the TCN, and the 
            output features are transposed back to the original format.
        """
        # if complex spectrum
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input
        feature = feature.transpose(1, 2)  # B, N, L

        feats = self.tcn(feature)  # [B, bottleneck_dim, L]
        feats = feats.transpose(1, 2)  # B, L, bottleneck_dim

        return feats, ilens

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_spk(self):
        """
        Number of speakers.

        This property returns the number of speakers for the separator. 
        In this implementation, it is set to None since the TCNSeparatorNomask
        does not estimate or require speaker information.

        Returns:
            None: Indicates that the number of speakers is not applicable for 
            this separator.

        Note:
            The value of this property is always None, reflecting the 
            design of the TCNSeparatorNomask class, which does not perform 
            speaker separation.

        Examples:
            >>> separator = TCNSeparatorNomask(input_dim=256)
            >>> separator.num_spk
            None
        """
        return None
