from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.skim import SkiM
from espnet2.enh.separator.abs_separator import AbsSeparator


class SkiMSeparator(AbsSeparator):
    """
    Skipping Memory (SkiM) Separator for speech separation tasks.

    This class implements the Skipping Memory (SkiM) Separator, which is designed
    for separating multiple speakers' audio from a mixed input. It utilizes 
    memory-augmented neural networks to enhance the separation quality. 

    Args:
        input_dim (int): Input feature dimension.
        causal (bool): Whether the system is causal. Default is True.
        num_spk (int): Number of target speakers. Default is 2.
        predict_noise (bool): Whether to predict noise in addition to speakers.
            Default is False.
        nonlinear (str): The nonlinear function for mask estimation. Must be 
            one of 'relu', 'tanh', or 'sigmoid'. Default is 'relu'.
        layer (int): Number of SkiM blocks. Default is 3.
        unit (int): Dimension of the hidden state. Default is 512.
        segment_size (int): Segmentation size for splitting long features.
            Default is 20.
        dropout (float): Dropout ratio. Default is 0.0.
        mem_type (str): Memory type for SegLSTM processing. Can be 'hc', 'h', 
            'c', 'id', or None. Default is 'hc'.
        seg_overlap (bool): Whether the segmentation will reserve 50% overlap 
            for adjacent segments. Default is False.

    Raises:
        ValueError: If an unsupported `mem_type` or `nonlinear` function is 
        provided.

    Examples:
        >>> separator = SkiMSeparator(input_dim=128, num_spk=2, layer=3)
        >>> input_tensor = torch.randn(10, 20, 128)  # (Batch, Time, Feature)
        >>> ilens = torch.tensor([20] * 10)  # Input lengths
        >>> masked, ilens, others = separator(input_tensor, ilens)
        >>> print(len(masked))  # Should be equal to num_spk

    Note:
        The `additional` argument in the `forward` method is not used in this 
        model but is kept for compatibility with the interface.

    Attributes:
        num_spk (int): The number of target speakers.
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.skim = SkiM(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
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
        Forward pass of the SkiMSeparator.

    This method processes the input features and returns the separated outputs 
    along with the input lengths and additional predicted data. The separation 
    is performed using the Skipping Memory (SkiM) mechanism.

    Args:
        input (Union[torch.Tensor, ComplexTensor]): Encoded feature tensor of 
            shape [B, T, N], where B is the batch size, T is the time dimension, 
            and N is the number of features.
        ilens (torch.Tensor): A tensor containing the lengths of the input 
            sequences, shape [Batch].
        additional (Optional[Dict], optional): Additional data included in the 
            model. Note that this parameter is not used in this model. Default 
            is None.

    Returns:
        Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, 
        OrderedDict]: A tuple containing:
            - masked (List[Union[torch.Tensor, ComplexTensor]]): A list of 
              tensors representing the masked outputs for each speaker, 
              each of shape [(B, T, N), ...].
            - ilens (torch.Tensor): A tensor containing the input lengths 
              of shape (B,).
            - others (OrderedDict): An ordered dictionary of additional 
              predicted data, including masks for each speaker:
                - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                - ...,
                - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

    Examples:
        >>> separator = SkiMSeparator(input_dim=128, num_spk=2)
        >>> input_tensor = torch.randn(10, 50, 128)  # [Batch, Time, Features]
        >>> ilens = torch.tensor([50] * 10)  # All sequences of length 50
        >>> masked, lengths, others = separator.forward(input_tensor, ilens)

    Note:
        The `additional` argument is included for compatibility with other 
        models but is not utilized in this implementation.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        processed = self.skim(feature)  # B,T, N

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

    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        """
        Process input frames in a streaming manner.

        This method performs the forward pass for streaming audio input, 
        allowing for real-time processing of audio frames. It computes 
        the masks for each target speaker based on the provided input 
        frames and maintains the state for continuous processing.

        Args:
            input_frame (torch.Tensor): Input audio frames with shape 
                [Batch, Time, Features].
            states (Optional): Optional states to maintain across 
                successive calls. Default is None.

        Returns:
            masked (List[Union[torch.Tensor, ComplexTensor]]): List of 
                masked audio frames for each target speaker.
            states: Updated states for subsequent calls.
            others (OrderedDict): Additional predicted data, such as 
                masks for each speaker:
                - 'mask_spk1': torch.Tensor(Batch, 1, Freq)
                - 'mask_spk2': torch.Tensor(Batch, 1, Freq)
                - ...
                - 'mask_spkn': torch.Tensor(Batch, 1, Freq)

        Examples:
            >>> separator = SkiMSeparator(input_dim=128)
            >>> input_frames = torch.randn(4, 10, 128)  # Batch of 4, 10 time steps
            >>> masked, states, others = separator.forward_streaming(input_frames)

        Note:
            This method is designed for use in scenarios where the 
            audio data is processed in small segments rather than all 
            at once. It is particularly useful for real-time applications.

        Raises:
            ValueError: If the input frame dimensions are incorrect.
        """
        if is_complex(input_frame):
            feature = abs(input_frame)
        else:
            feature = input_frame

        B, _, N = feature.shape

        processed, states = self.skim.forward_stream(feature, states=states)

        processed = processed.view(B, 1, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input_frame * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input_frame * mask_noise

        return masked, states, others

    @property
    def num_spk(self):
        return self._num_spk
