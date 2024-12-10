from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import TemporalConvNet
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class TCNSeparator(AbsSeparator):
    """
        TCNSeparator is a temporal convolutional network-based separator for speech
    enhancement. It utilizes a TemporalConvNet to separate audio signals from
    multiple speakers, optionally predicting noise in the process.

    Attributes:
        num_spk (int): Number of speakers in the input audio.
        predict_noise (bool): Flag to indicate whether to output the estimated
            noise signal.
        masking (bool): Flag to choose between masking and mapping-based methods.

    Args:
        input_dim (int): Input feature dimension.
        num_spk (int, optional): Number of speakers (default is 2).
        predict_noise (bool, optional): Whether to output the estimated noise
            signal (default is False).
        layer (int, optional): Number of layers in each stack (default is 8).
        stack (int, optional): Number of stacks (default is 3).
        bottleneck_dim (int, optional): Bottleneck dimension (default is 128).
        hidden_dim (int, optional): Number of convolution channels (default is 512).
        kernel (int, optional): Kernel size (default is 3).
        causal (bool, optional): Whether to use causal convolutions (default is
            False).
        norm_type (str, optional): Normalization type, choose from 'BN', 'gLN',
            'cLN' (default is 'gLN').
        nonlinear (str, optional): Nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid', 'linear' (default is 'relu').
        pre_mask_nonlinear (str, optional): Non-linear function before masknet
            (default is 'prelu').
        masking (bool, optional): Whether to use the masking or mapping based
            method (default is True).

    Raises:
        ValueError: If the specified nonlinear function is not supported.

    Examples:
        # Initialize TCNSeparator
        separator = TCNSeparator(input_dim=256, num_spk=2)

        # Forward pass with input tensor and lengths
        input_tensor = torch.randn(4, 100, 256)  # [Batch, Time, Feature]
        ilens = torch.tensor([100, 100, 100, 100])  # input lengths
        masked, ilens, others = separator(input_tensor, ilens)

        # Forward streaming
        streaming_output, buffer, others_streaming = separator.forward_streaming(
            input_frame=torch.randn(4, 1, 256)  # [Batch, 1, Feature]
        )
    """

    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        predict_noise: bool = False,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
        nonlinear: str = "relu",
        pre_mask_nonlinear: str = "prelu",
        masking: bool = True,
    ):
        """Temporal Convolution Separator

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            layer: int, number of layers in each stack.
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid', 'linear'
            pre_mask_nonlinear: the non-linear function before masknet
            masking: whether to use the masking or mapping based method
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        if nonlinear not in ("sigmoid", "relu", "tanh", "linear"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.tcn = TemporalConvNet(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            C=num_spk + 1 if predict_noise else num_spk,
            norm_type=norm_type,
            causal=causal,
            pre_mask_nonlinear=pre_mask_nonlinear,
            mask_nonlinear=nonlinear,
        )

        self.masking = masking

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
                Forward pass for the Temporal Convolution Separator.

        This method processes the input features through the Temporal Convolution Network
        (TCN) to generate masked output signals for each speaker, along with additional
        predicted data, such as noise estimates if enabled.

        Args:
            input (Union[torch.Tensor, ComplexTensor]): Encoded feature of shape [B, T, N],
                where B is the batch size, T is the time dimension, and N is the number of
                features. The input can be a real tensor or a complex tensor.
            ilens (torch.Tensor): Input lengths of shape [Batch], indicating the actual
                lengths of each input sequence.
            additional (Optional[Dict]): Additional data included in the model, which is
                currently not used in this implementation. Defaults to None.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
                - masked (List[Union[torch.Tensor, ComplexTensor]]): List of tensors,
                  each representing the masked output for the speakers, shape [(B, T, N), ...].
                - ilens (torch.Tensor): Input lengths of shape (B,), as passed.
                - others (OrderedDict): Dictionary containing other predicted data, e.g. masks:
                  OrderedDict[
                      'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                      'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                      ...
                      'mask_spkn': torch.Tensor(Batch, Frames, Freq),
                  ]

        Examples:
            # Example of using the forward method
            separator = TCNSeparator(input_dim=128, num_spk=2)
            input_tensor = torch.randn(10, 50, 128)  # Example input
            ilens = torch.tensor([50] * 10)  # Example lengths
            masked, ilens, others = separator.forward(input_tensor, ilens)

        Note:
            Ensure that the input dimensions and types are consistent with the expected
            shapes and types as outlined in the arguments.

        Raises:
            ValueError: If the input nonlinear activation function is not one of the
            supported types: 'sigmoid', 'relu', 'tanh', or 'linear'.
        """
        # if complex spectrum
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input
        B, L, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, L

        masks = self.tcn(feature)  # B, num_spk, N, L
        masks = masks.transpose(2, 3)  # B, num_spk, L, N
        if self.predict_noise:
            *masks, mask_noise = masks.unbind(dim=1)  # List[B, L, N]
        else:
            masks = masks.unbind(dim=1)  # List[B, L, N]

        if self.masking:
            # masking-based SE
            masked = [input * m for m in masks]
        else:
            # mapping-based SE
            masked = [m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    def forward_streaming(self, input_frame: torch.Tensor, buffer=None):
        """
        Forward streaming for the Temporal Convolution Separator.

        This method processes an input frame for real-time audio separation.
        It maintains a buffer to accommodate the temporal context required
        by the temporal convolution network. The input is rolled into the
        buffer to simulate streaming input.

        Args:
            input_frame (torch.Tensor): The input audio frame of shape
                (B, 1, N) where B is the batch size and N is the number of
                features.
            buffer (torch.Tensor, optional): The buffer containing past
                frames of shape (B, receptive_field, N). If None, a new
                buffer will be initialized.

        Returns:
            Tuple[List[Union[torch.Tensor, ComplexTensor]],
                  torch.Tensor, OrderedDict]:
                - masked (List[Union[torch.Tensor, ComplexTensor]]):
                    List of masked output tensors for each speaker.
                - buffer (torch.Tensor): The updated buffer after
                    processing the input frame.
                - others (OrderedDict): Additional outputs such as
                    masks for each speaker.

        Examples:
            >>> separator = TCNSeparator(input_dim=64, num_spk=2)
            >>> input_frame = torch.randn(4, 1, 64)  # Batch size of 4
            >>> masked_output, updated_buffer, additional_outputs =
            ... separator.forward_streaming(input_frame)

        Note:
            This method requires that the TemporalConvNet has been
            properly initialized and the input frame has the correct shape.

        Raises:
            ValueError: If the input_frame does not have the correct shape.
        """
        # input_frame: B, 1, N

        B, _, N = input_frame.shape

        receptive_field = self.tcn.receptive_field

        if buffer is None:
            buffer = torch.zeros((B, receptive_field, N), device=input_frame.device)

        buffer = torch.roll(buffer, shifts=-1, dims=1)
        buffer[:, -1, :] = input_frame[:, 0, :]

        masked, ilens, others = self.forward(buffer, None)

        masked = [m[:, -1, :].unsqueeze(1) for m in masked]

        return masked, buffer, others

    @property
    def num_spk(self):
        return self._num_spk
