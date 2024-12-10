from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.dptnet import DPTNet
from espnet2.enh.layers.tcn import choose_norm
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class DPTNetSeparator(AbsSeparator):
    """
    Dual-Path Transformer Network (DPTNet) Separator for audio source separation.

This class implements the DPTNet architecture for separating audio sources 
based on input features. It utilizes a dual-path strategy to efficiently 
process audio signals and estimate the masks for multiple speakers.

Attributes:
    num_spk (int): The number of speakers for separation.
    predict_noise (bool): Indicates if the estimated noise signal should be 
        output.
    segment_size (int): The size of the segments used in dual-path processing.
    post_enc_relu (bool): If True, applies ReLU activation after encoding.
    enc_LN: Normalization layer applied after encoding.
    num_outputs (int): The number of outputs, including the estimated noise 
        if applicable.
    dptnet: The DPTNet model instance used for processing.
    output: The gated output layer for generating filters.
    output_gate: The gate layer for controlling output activation.
    nonlinear: The nonlinear function used for mask estimation.

Args:
    input_dim (int): Input feature dimension.
    post_enc_relu (bool): If True, applies ReLU after encoding. Default is True.
    rnn_type (str): Select from 'RNN', 'LSTM', or 'GRU'. Default is 'lstm'.
    bidirectional (bool): Whether inter-chunk RNN layers are bidirectional. 
        Default is True.
    num_spk (int): Number of speakers. Default is 2.
    predict_noise (bool): Whether to output the estimated noise signal. 
        Default is False.
    unit (int): Dimension of the hidden state. Default is 256.
    att_heads (int): Number of attention heads. Default is 4.
    dropout (float): Dropout ratio. Default is 0.0.
    activation (str): Activation function applied at the output of RNN. 
        Default is 'relu'.
    norm_type (str): Type of normalization to use after Transformer blocks. 
        Default is 'gLN'.
    layer (int): Number of stacked RNN layers. Default is 6.
    segment_size (int): Dual-path segment size. Default is 20.
    nonlinear (str): Nonlinear function for mask estimation, 
        select from 'relu', 'tanh', 'sigmoid'. Default is 'relu'.

Raises:
    ValueError: If `nonlinear` is not one of 'sigmoid', 'relu', or 'tanh'.

Examples:
    # Initialize the DPTNetSeparator
    separator = DPTNetSeparator(input_dim=256, num_spk=2, predict_noise=True)

    # Forward pass through the separator
    masked, ilens, others = separator.forward(input_tensor, input_lengths)

    # Access the estimated masks
    mask_spk1 = others['mask_spk1']
    mask_spk2 = others['mask_spk2']
    noise_estimate = others.get('noise1', None)
    """
    def __init__(
        self,
        input_dim: int,
        post_enc_relu: bool = True,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        unit: int = 256,
        att_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "gLN",
        layer: int = 6,
        segment_size: int = 20,
        nonlinear: str = "relu",
    ):
        """Dual-Path Transformer Network (DPTNet) Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            unit: int, dimension of the hidden state.
            att_heads: number of attention heads.
            dropout: float, dropout ratio. Default is 0.
            activation: activation function applied at the output of RNN.
            norm_type: type of normalization to use after each inter- or
                intra-chunk Transformer block.
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            segment_size: dual-path segment size
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.segment_size = segment_size

        self.post_enc_relu = post_enc_relu
        self.enc_LN = choose_norm(norm_type, input_dim)
        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.dptnet = DPTNet(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            att_heads=att_heads,
            dropout=dropout,
            activation=activation,
            num_layers=layer,
            bidirectional=bidirectional,
            norm_type=norm_type,
        )
        # gated output layer
        self.output = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, 1), torch.nn.Tanh()
        )
        self.output_gate = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, 1), torch.nn.Sigmoid()
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
        Forward pass for the DPTNetSeparator.

    This method processes the input features through the DPTNet architecture,
    applying necessary transformations and returning the masked outputs along
    with the predicted masks for each speaker.

    Args:
        input (Union[torch.Tensor, ComplexTensor]): Encoded feature of shape
            [B, T, N], where B is the batch size, T is the time frames, and N
            is the feature dimension.
        ilens (torch.Tensor): Input lengths of shape [Batch].
        additional (Optional[Dict]): Other data included in the model.
            NOTE: This parameter is not used in this model.

    Returns:
        Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
            - masked (List[Union[torch.Tensor, ComplexTensor]]): List of masked
              outputs, each of shape [(B, T, N), ...].
            - ilens (torch.Tensor): Tensor of input lengths with shape (B,).
            - others (OrderedDict): Dictionary containing predicted data, e.g. masks:
                - 'mask_spk1': torch.Tensor of shape (Batch, Frames, Freq),
                - 'mask_spk2': torch.Tensor of shape (Batch, Frames, Freq),
                ...
                - 'mask_spkn': torch.Tensor of shape (Batch, Frames, Freq).

    Examples:
        >>> separator = DPTNetSeparator(input_dim=128)
        >>> input_tensor = torch.randn(10, 100, 128)  # Batch of 10
        >>> ilens = torch.tensor([100] * 10)  # All sequences of length 100
        >>> masked, lengths, others = separator(input_tensor, ilens)

    Note:
        This method is designed to handle both real and complex input tensors.
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        elif self.post_enc_relu:
            feature = torch.nn.functional.relu(input)
        else:
            feature = input

        B, T, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, T
        feature = self.enc_LN(feature)
        segmented = self.split_feature(feature)  # B, N, L, K

        processed = self.dptnet(segmented)  # B, N*num_spk, L, K
        processed = processed.reshape(
            B * self.num_outputs, -1, processed.size(-2), processed.size(-1)
        )  # B*num_spk, N, L, K

        processed = self.merge_feature(processed, length=T)  # B*num_spk, N, T

        # gated output layer for filter generation (B*num_spk, N, T)
        processed = self.output(processed) * self.output_gate(processed)

        masks = processed.reshape(B, self.num_outputs, N, T)

        # list[(B, T, N)]
        masks = self.nonlinear(masks.transpose(-1, -2)).unbind(dim=1)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    def split_feature(self, x):
        """
        Dual-Path Transformer Network (DPTNet) Separator.

        This class implements a DPTNet separator for audio source separation
        tasks. It leverages a dual-path architecture that processes audio 
        features for multiple speakers, optionally estimating noise signals.

        Attributes:
            _num_spk (int): Number of speakers.
            predict_noise (bool): Whether to output the estimated noise signal.
            segment_size (int): Dual-path segment size.
            post_enc_relu (bool): Apply ReLU after encoding.
            enc_LN: Normalization layer.
            num_outputs (int): Number of outputs (speakers + noise).
            dptnet: Instance of the DPTNet class for processing.
            output: Gated output layer for filter generation.
            output_gate: Gated output layer for controlling output.
            nonlinear: Nonlinear activation function for mask estimation.

        Args:
            input_dim (int): Input feature dimension.
            post_enc_relu (bool): If True, apply ReLU after encoding.
            rnn_type (str): Type of RNN ('RNN', 'LSTM', 'GRU').
            bidirectional (bool): If True, use bidirectional RNN layers.
            num_spk (int): Number of speakers to separate.
            predict_noise (bool): If True, output the estimated noise signal.
            unit (int): Dimension of the hidden state.
            att_heads (int): Number of attention heads.
            dropout (float): Dropout ratio. Default is 0.
            activation (str): Activation function applied at RNN output.
            norm_type (str): Type of normalization to use.
            layer (int): Number of stacked RNN layers. Default is 3.
            segment_size (int): Size of each segment in dual-path processing.
            nonlinear (str): Nonlinear function for mask estimation ('relu', 
                             'tanh', 'sigmoid').

        Raises:
            ValueError: If an unsupported nonlinear function is provided.

        Examples:
            >>> separator = DPTNetSeparator(input_dim=256, num_spk=2)
            >>> input_tensor = torch.randn(10, 20, 256)  # Batch, Time, Feature
            >>> ilens = torch.tensor([20] * 10)  # Input lengths
            >>> masked, ilens, others = separator(input_tensor, ilens)
        """
        B, N, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, N, self.segment_size, -1)

    def merge_feature(self, x, length=None):
        """
        Merge feature chunks back into a single feature sequence.

    This method takes the output of the dual-path processing and merges the
    feature chunks into a single sequence using a folding operation. It 
    handles both cases where the output length is specified or needs to be 
    inferred from the number of chunks.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, L, n_chunks) where:
            B - batch size,
            N - number of feature channels,
            L - length of each feature chunk,
            n_chunks - number of chunks to merge.
        length (Optional[int]): Desired length of the output sequence. If 
            None, the length is calculated based on the number of chunks 
            and segment size.

    Returns:
        torch.Tensor: Merged feature tensor of shape (B, N, length).

    Note:
        The output is normalized by the number of overlapping segments used 
        during the merge process.

    Examples:
        >>> separator = DPTNetSeparator(input_dim=128)
        >>> x = torch.randn(2, 64, 10, 4)  # Example input
        >>> merged_features = separator.merge_feature(x, length=40)
        >>> print(merged_features.shape)  # Output: torch.Size([2, 64, 40])

    Raises:
        ValueError: If the input tensor `x` does not have the expected 
        dimensions.
        """
        B, N, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, N * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, N, length)

    @property
    def num_spk(self):
        return self._num_spk
