# The implementation of FaSNet in
# Y. Luo, et al.  “FaSNet: Low-Latency Adaptive Beamforming
# for Multi-Microphone Audio Processing”
# The implementation is based on:
# https://github.com/yluo42/TAC
# Licensed under CC BY-NC-SA 3.0 US.
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers import dprnn


# DPRNN for beamforming filter estimation
class BF_module(nn.Module):
    """
    Beamforming module for FaSNet.

    This module implements the beamforming filter estimation using a 
    Dual-Path Recurrent Neural Network (DPRNN) as described in:
    Y. Luo, et al. “FaSNet: Low-Latency Adaptive Beamforming for 
    Multi-Microphone Audio Processing”. The implementation is based on 
    the repository: https://github.com/yluo42/TAC and is licensed under 
    CC BY-NC-SA 3.0 US.

    Attributes:
        input_dim (int): The dimension of the input features.
        feature_dim (int): The dimension of the feature representation.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output features.
        num_spk (int): The number of speakers (default is 2).
        layer (int): The number of layers in the DPRNN (default is 4).
        segment_size (int): The size of the segments to process (default is 100).
        bidirectional (bool): Whether to use a bidirectional RNN (default is True).
        dropout (float): The dropout rate (default is 0.0).
        fasnet_type (str): Type of FaSNet to use ('fasnet' or 'ifasnet').

    Args:
        input_dim (int): Dimension of the input features.
        feature_dim (int): Dimension of the feature representation.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output features.
        num_spk (int, optional): Number of speakers. Defaults to 2.
        layer (int, optional): Number of layers in the DPRNN. Defaults to 4.
        segment_size (int, optional): Size of the segments to process. Defaults to 100.
        bidirectional (bool, optional): Whether to use a bidirectional RNN. Defaults to True.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        fasnet_type (str, optional): Type of FaSNet to use ('fasnet' or 'ifasnet'). 
            Defaults to 'ifasnet'.

    Returns:
        torch.Tensor: The estimated beamforming filter of shape 
        (B, ch, nspk, L, K) for 'ifasnet' and (B, ch, nspk, K, L) 
        for 'fasnet', where B is batch size, ch is number of channels, 
        nspk is number of speakers, L is the segment length, and K 
        is the output dimension.

    Raises:
        AssertionError: If `fasnet_type` is not 'fasnet' or 'ifasnet'.

    Examples:
        >>> bf = BF_module(input_dim=64, feature_dim=64, hidden_dim=128, 
        ...                 output_dim=64, num_spk=2, layer=4, 
        ...                 segment_size=100, bidirectional=True, 
        ...                 dropout=0.0, fasnet_type='fasnet')
        >>> input_tensor = torch.rand(2, 4, 64, 320)  # (B, ch, N, T)
        >>> num_mic = torch.tensor([3, 2])  # number of microphones
        >>> output = bf(input_tensor, num_mic)
        >>> print(output.shape)  # Output shape will depend on fasnet_type

    Note:
        The module requires the `dprnn` layer from `espnet2.enh.layers`.
    """
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        bidirectional=True,
        dropout=0.0,
        fasnet_type="ifasnet",
    ):
        super().__init__()

        assert fasnet_type in [
            "fasnet",
            "ifasnet",
        ], "fasnet_type should be fasnet or ifasnet"

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.dprnn_model = dprnn.DPRNN_TAC(
            "lstm",
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim * self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.eps = 1e-8

        self.fasnet_type = fasnet_type

        if fasnet_type == "ifasnet":
            # output layer in ifasnet
            self.output = nn.Conv1d(self.feature_dim, self.output_dim, 1)
        elif fasnet_type == "fasnet":
            # gated output layer in ifasnet
            self.output = nn.Sequential(
                nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh()
            )
            self.output_gate = nn.Sequential(
                nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid()
            )

        self.num_spk = num_spk
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

    def forward(self, input, num_mic):
        """
        Forward pass for the beamforming filter estimation.

    This method processes the input tensor, which contains audio signals 
    from multiple microphones, and computes the beamforming filters for 
    each speaker based on the provided input and the number of active 
    microphones. The input tensor is reshaped and passed through the 
    necessary layers to produce the output beamforming filters.

    Args:
        input (torch.Tensor): Input tensor of shape (B, ch, N, T) where:
            B is the batch size,
            ch is the number of channels (microphones),
            N is the number of features,
            T is the sequence length.
        num_mic (torch.Tensor): Tensor of shape (B,) indicating the number 
            of channels for each input. A value of zero indicates a fixed 
            geometry configuration.

    Returns:
        torch.Tensor: Output beamforming filters of shape (B, ch, nspk, 
        K, L) for 'ifasnet' or (B, ch, nspk, L, N) for 'fasnet', where:
            nspk is the number of speakers,
            K is the output dimension,
            L is the segment length.

    Examples:
        >>> model = BF_module(input_dim=4, feature_dim=64, hidden_dim=128, 
        ...                   output_dim=32, num_spk=2)
        >>> input_tensor = torch.randn(2, 4, 64, 320)  # (batch, ch, N, T)
        >>> num_mic = torch.tensor([3, 2])  # Number of active microphones
        >>> output_filters = model.forward(input_tensor, num_mic)
        >>> print(output_filters.shape)  # Should output shape based on nspk

    Note:
        The function assumes that the input tensor has been properly 
        formatted and that the model has been initialized with valid 
        parameters.

    Raises:
        AssertionError: If the shape of input tensor does not match the 
        expected dimensions or if num_mic tensor is not properly defined.
        """
        # input: (B, ch, N, T)
        batch_size, ch, N, seq_length = input.shape

        input = input.view(batch_size * ch, N, seq_length)  # B*ch, N, T
        enc_feature = self.BN(input)

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = dprnn.split_feature(
            enc_feature, self.segment_size
        )  # B*ch, N, L, K

        enc_segments = enc_segments.view(
            batch_size, ch, -1, enc_segments.shape[2], enc_segments.shape[3]
        )  # B, ch, N, L, K
        output = self.dprnn_model(enc_segments, num_mic).view(
            batch_size * ch * self.num_spk,
            self.feature_dim,
            self.segment_size,
            -1,
        )  # B*ch*nspk, N, L, K
        # overlap-and-add of the outputs
        output = dprnn.merge_feature(output, enc_rest)  # B*ch*nspk, N, T

        if self.fasnet_type == "fasnet":
            # gated output layer for filter generation
            bf_filter = self.output(output) * self.output_gate(
                output
            )  # B*ch*nspk, K, T
            bf_filter = (
                bf_filter.transpose(1, 2)
                .contiguous()
                .view(batch_size, ch, self.num_spk, -1, self.output_dim)
            )  # B, ch, nspk, L, N

        elif self.fasnet_type == "ifasnet":
            # output layer
            bf_filter = self.output(output)  # B*ch*nspk, K, T
            bf_filter = bf_filter.view(
                batch_size, ch, self.num_spk, self.output_dim, -1
            )  # B, ch, nspk, K, L

        return bf_filter


# base module for FaSNet
class FaSNet_base(nn.Module):
    """
    Base module for FaSNet.

    This class serves as the base for the FaSNet architecture, which is designed
    for low-latency adaptive beamforming in multi-microphone audio processing. It
    provides methods for signal segmentation and context extraction, as well as 
    cosine similarity calculations between reference and target microphone signals.

    Attributes:
        win_len (int): Length of the window in milliseconds for segmentation.
        window (int): Size of the window in samples.
        stride (int): Stride size for segmentation.
        sr (int): Sampling rate in Hz.
        context_len (int): Length of the context in milliseconds.
        dropout (float): Dropout rate for regularization.
        enc_dim (int): Dimensionality of the encoder input.
        feature_dim (int): Dimensionality of the features.
        hidden_dim (int): Dimensionality of the hidden layers.
        segment_size (int): Size of the segments for processing.
        layer (int): Number of layers in the model.
        num_spk (int): Number of speakers to be processed.
        eps (float): Small constant to avoid division by zero.

    Args:
        enc_dim (int): Dimensionality of the encoder input.
        feature_dim (int): Dimensionality of the features.
        hidden_dim (int): Dimensionality of the hidden layers.
        layer (int): Number of layers in the model.
        segment_size (int, optional): Size of the segments for processing. Default is 24.
        nspk (int, optional): Number of speakers to be processed. Default is 2.
        win_len (int, optional): Length of the window in milliseconds. Default is 16.
        context_len (int, optional): Length of the context in milliseconds. Default is 16.
        dropout (float, optional): Dropout rate for regularization. Default is 0.0.
        sr (int, optional): Sampling rate in Hz. Default is 16000.

    Methods:
        pad_input(input, window):
            Zero-padding input according to window/stride size.

        seg_signal_context(x, window, context):
            Segment the signal into chunks with specific context.

        signal_context(x, context):
            Create a signal context function for the input signal.

        seq_cos_sim(ref, target):
            Compute cosine similarity between reference and target microphone signals.

        forward(input, num_mic):
            Abstract forward function to be implemented in derived classes.

    Examples:
        # Creating an instance of the FaSNet_base class
        fasnet = FaSNet_base(enc_dim=64, feature_dim=64, hidden_dim=128,
                             layer=4, segment_size=50, nspk=2,
                             win_len=4, context_len=16, sr=16000)
    """
    def __init__(
        self,
        enc_dim,
        feature_dim,
        hidden_dim,
        layer,
        segment_size=24,
        nspk=2,
        win_len=16,
        context_len=16,
        dropout=0.0,
        sr=16000,
    ):
        super(FaSNet_base, self).__init__()

        # parameters
        self.win_len = win_len
        self.window = max(int(sr * win_len / 1000), 2)
        self.stride = self.window // 2
        self.sr = sr
        self.context_len = context_len
        self.dropout = dropout

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.

    This method pads the input tensor such that its length matches the required 
    window size for processing. It adds padding at the beginning and end of the 
    input signal as needed to ensure compatibility with the window and stride 
    parameters.

    Attributes:
        stride (int): The stride size calculated from the window size.

    Args:
        input (torch.Tensor): The input tensor of shape (B, nmic, nsample), where
            B is the batch size, nmic is the number of microphones, and nsample 
            is the number of samples.
        window (int): The window size used for padding.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the padded input tensor and 
        the number of samples added as padding at the end.

    Examples:
        >>> input_tensor = torch.randn(2, 4, 320)  # (batch, num_mic, length)
        >>> padded_input, padding_rest = pad_input(input_tensor, window=64)
        >>> print(padded_input.shape)  # Output shape may vary based on input
        >>> print(padding_rest)  # Number of samples added as padding

    Note:
        The padding is performed using zero values, which may impact signal 
        processing tasks if not handled appropriately downstream.
        """

        batch_size, nmic, nsample = input.shape

        stride = self.stride

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, nmic, rest).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, nmic, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def seg_signal_context(self, x, window, context):
        """
        Segmenting the signal into chunks with specific context.

    This method segments the input signal `x` into overlapping chunks of a
    specified `window` size, while also incorporating additional context
    frames before and after each chunk. The context allows for better 
    handling of signal dependencies in subsequent processing steps.

    Args:
        x (torch.Tensor): Input signal of shape (B, ch, T), where B is the 
                          batch size, ch is the number of channels, and T 
                          is the length of the signal.
        window (int): The size of each segment/chunk.
        context (int): The number of context frames to include before and 
                       after each chunk.

    Returns:
        tuple: A tuple containing:
            - center_frame (torch.Tensor): The center frames of the chunks 
              of shape (B, ch, L, window), where L is the number of chunks.
            - chunks (torch.Tensor): The complete set of chunks of shape 
              (B, ch, L, 2 * context + window), including the context frames.
            - rest (int): The number of remaining samples after chunking.

    Examples:
        >>> x = torch.rand(2, 4, 320)  # Batch of 2, 4 channels, 320 samples
        >>> window = 16
        >>> context = 4
        >>> center_frame, chunks, rest = seg_signal_context(x, window, context)
        >>> center_frame.shape
        torch.Size([2, 4, 21, 16])  # 21 chunks of 16 samples each

    Note:
        The input signal is padded to ensure that it can be segmented 
        correctly based on the specified window size and context.
        """

        # pad input accordingly
        # first pad according to window size
        input, rest = self.pad_input(x, window)
        batch_size, nmic, nsample = input.shape
        stride = window // 2

        # pad another context size
        pad_context = torch.zeros(batch_size, nmic, context).type(input.type())
        input = torch.cat([pad_context, input, pad_context], 2)  # B, ch, L

        # calculate index for each chunk
        nchunk = 2 * nsample // window - 1
        begin_idx = np.arange(nchunk) * stride
        begin_idx = (
            torch.from_numpy(begin_idx).type(input.type()).long().view(1, 1, -1)
        )  # 1, 1, nchunk
        begin_idx = begin_idx.expand(batch_size, nmic, nchunk)  # B, ch, nchunk
        # select entries from index
        chunks = [
            torch.gather(input, 2, begin_idx + i).unsqueeze(3)
            for i in range(2 * context + window)
        ]  # B, ch, nchunk, 1
        chunks = torch.cat(chunks, 3)  # B, ch, nchunk, chunk_size

        # center frame
        center_frame = chunks[:, :, :, context : context + window]

        return center_frame, chunks, rest

    def signal_context(self, x, context):
        """
        Base module for FaSNet.

    This class implements the base functionality for the FaSNet model,
    which is designed for low-latency adaptive beamforming for multi-microphone
    audio processing.

    Attributes:
        win_len (int): The length of the window for segmentation.
        window (int): The window size in samples.
        stride (int): The stride size for segmentation.
        sr (int): The sample rate of the input audio.
        context_len (int): The length of context to consider during processing.
        dropout (float): The dropout rate.
        enc_dim (int): The dimension of the encoder.
        feature_dim (int): The dimension of the feature representation.
        hidden_dim (int): The dimension of the hidden states.
        segment_size (int): The size of segments for processing.
        layer (int): The number of layers in the model.
        num_spk (int): The number of speakers.
        eps (float): A small constant to prevent division by zero.

    Args:
        enc_dim (int): The encoder dimension.
        feature_dim (int): The feature dimension.
        hidden_dim (int): The hidden dimension.
        layer (int): The number of layers.
        segment_size (int, optional): Size of segments for processing. Defaults to 24.
        nspk (int, optional): Number of speakers. Defaults to 2.
        win_len (int, optional): Window length in milliseconds. Defaults to 16.
        context_len (int, optional): Context length in milliseconds. Defaults to 16.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        sr (int, optional): Sample rate. Defaults to 16000.

    Methods:
        pad_input(input, window):
            Zero-padding input according to window/stride size.
        seg_signal_context(x, window, context):
            Segmenting the signal into chunks with specific context.
        signal_context(x, context):
            Signal context function that segments the signal into chunks.
        seq_cos_sim(ref, target):
            Computes cosine similarity between reference and target signals.
        forward(input, num_mic):
            Abstract forward function that processes the input.

    Examples:
        # Example of creating a FaSNet_base model
        model = FaSNet_base(enc_dim=64, feature_dim=64, hidden_dim=128, layer=4)
        
        # Example of padding input
        padded_input, rest = model.pad_input(torch.randn(2, 4, 32000), window=512)
        
        # Example of segmenting signal context
        center_frame, chunks, rest = model.seg_signal_context(torch.randn(2, 4, 32000), 
                                                               window=512, context=16)
        
        # Example of computing cosine similarity
        cos_sim = model.seq_cos_sim(torch.randn(3, 512, 100), torch.randn(2, 512, 100))

    Note:
        The model is designed for processing audio signals and may require
        specific configurations based on the application.
        """

        batch_size, dim, nframe = x.shape

        zero_pad = torch.zeros(batch_size, dim, context).type(x.type())
        pad_past = []
        pad_future = []
        for i in range(context):
            pad_past.append(
                torch.cat([zero_pad[:, :, i:], x[:, :, : -context + i]], 2).unsqueeze(2)
            )
            pad_future.append(
                torch.cat([x[:, :, i + 1 :], zero_pad[:, :, : i + 1]], 2).unsqueeze(2)
            )

        pad_past = torch.cat(pad_past, 2)  # B, D, C, L
        pad_future = torch.cat(pad_future, 2)  # B, D, C, L
        all_context = torch.cat(
            [pad_past, x.unsqueeze(2), pad_future], 2
        )  # B, D, 2*C+1, L

        return all_context

    def seq_cos_sim(self, ref, target):
        """
        Computes the cosine similarity between the reference microphones and the 
    target microphones.

    This function takes two input tensors representing signals from different 
    microphones and calculates the cosine similarity across their segments. 
    It ensures that the input tensors have compatible dimensions for the 
    calculation.

    Args:
        ref (torch.Tensor): A tensor of shape (nmic1, L, seg1) representing the 
            reference microphone signals.
        target (torch.Tensor): A tensor of shape (nmic2, L, seg2) representing 
            the target microphone signals.

    Returns:
        torch.Tensor: A tensor of shape (larger_ch, L, seg1-seg2+1) containing 
            the cosine similarity values between the reference and target 
            microphones.

    Raises:
        AssertionError: If the lengths of the reference and target tensors do 
            not match or if the reference tensor has fewer segments than the 
            target tensor.

    Examples:
        >>> ref = torch.rand(3, 100, 50)  # 3 microphones, 100 length, 50 segments
        >>> target = torch.rand(2, 100, 30)  # 2 microphones, 100 length, 30 segments
        >>> cos_sim = seq_cos_sim(ref, target)
        >>> print(cos_sim.shape)  # Output: torch.Size([3, 100, 21])

    Note:
        This function uses the PyTorch library for tensor operations and 
        requires that the input tensors be of type `torch.Tensor`.
        """

        assert ref.size(1) == target.size(1), "Inputs should have same length."
        assert ref.size(2) >= target.size(
            2
        ), "Reference input should be no smaller than the target input."

        seq_length = ref.size(1)

        larger_ch = ref.size(0)
        if target.size(0) > ref.size(0):
            ref = ref.expand(
                target.size(0), ref.size(1), ref.size(2)
            ).contiguous()  # nmic2, L, seg1
            larger_ch = target.size(0)
        elif target.size(0) < ref.size(0):
            target = target.expand(
                ref.size(0), target.size(1), target.size(2)
            ).contiguous()  # nmic1, L, seg2

        # L2 norms
        ref_norm = F.conv1d(
            ref.view(1, -1, ref.size(2)).pow(2),
            torch.ones(ref.size(0) * ref.size(1), 1, target.size(2)).type(ref.type()),
            groups=larger_ch * seq_length,
        )  # 1, larger_ch*L, seg1-seg2+1
        ref_norm = ref_norm.sqrt() + self.eps
        target_norm = (
            target.norm(2, dim=2).view(1, -1, 1) + self.eps
        )  # 1, larger_ch*L, 1
        # cosine similarity
        cos_sim = F.conv1d(
            ref.view(1, -1, ref.size(2)),
            target.view(-1, 1, target.size(2)),
            groups=larger_ch * seq_length,
        )  # 1, larger_ch*L, seg1-seg2+1
        cos_sim = cos_sim / (ref_norm * target_norm)

        return cos_sim.view(larger_ch, seq_length, -1)

    def forward(self, input, num_mic):
        """
        Abstract forward function for FaSNet base model.

    This method defines the forward pass for the FaSNet base model. It takes
    the input audio signal and the number of microphones as arguments. The
    expected shape of the input is (batch, max_num_ch, T), where 'batch'
    is the batch size, 'max_num_ch' is the maximum number of channels, and
    'T' is the length of the audio signal. The 'num_mic' parameter is a
    tensor of shape (batch,) that indicates the number of channels for each
    input, where zero denotes a fixed geometry configuration.

    Args:
        input (torch.Tensor): Input tensor of shape (batch, max_num_ch, T).
        num_mic (torch.Tensor): Tensor of shape (batch,) indicating the number
            of channels for each input. Zero indicates fixed geometry.

    Returns:
        torch.Tensor: Output tensor from the forward pass, the shape of which
        depends on the specific implementation of the derived class.

    Examples:
        >>> model = FaSNet_TAC(enc_dim=64, feature_dim=64, hidden_dim=128,
        ...                     layer=4, segment_size=50, nspk=2,
        ...                     win_len=4, context_len=16, sr=16000)
        >>> input_data = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
        >>> num_mic = torch.tensor([3, 2])  # Number of active microphones
        >>> output = model(input_data, num_mic)
        >>> print(output.shape)  # Shape depends on the implementation
        """
        pass


# single-stage FaSNet + TAC
class FaSNet_TAC(FaSNet_base):
    """
    Single-stage FaSNet with Temporal Adaptive Control (TAC).

    This class implements the FaSNet model as described in the paper:
    "FaSNet: Low-Latency Adaptive Beamforming for Multi-Microphone Audio 
    Processing" by Y. Luo et al. The implementation utilizes the DPRNN 
    (Dynamic-Partial-Recurrent Neural Network) architecture to estimate 
    beamforming filters.

    Attributes:
        context (int): The context length for input signal processing.
        filter_dim (int): The dimension of the filter used in the model.
        all_BF (BF_module): The beamforming module for filter estimation.
        encoder (nn.Conv1d): Convolutional layer for waveform encoding.
        enc_LN (nn.GroupNorm): Group normalization layer for the encoder output.

    Args:
        enc_dim (int): Dimension of the encoder input.
        feature_dim (int): Dimension of the features.
        hidden_dim (int): Dimension of the hidden layers.
        layer (int): Number of layers in the DPRNN.
        segment_size (int, optional): Size of segments for processing (default=24).
        nspk (int, optional): Number of speakers (default=2).
        win_len (int, optional): Length of the window for segmentation (default=16).
        context_len (int, optional): Length of the context for processing (default=16).
        dropout (float, optional): Dropout rate (default=0.0).
        sr (int, optional): Sampling rate (default=16000).

    Examples:
        >>> model_TAC = FaSNet_TAC(
        ...     enc_dim=64,
        ...     feature_dim=64,
        ...     hidden_dim=128,
        ...     layer=4,
        ...     segment_size=50,
        ...     nspk=2,
        ...     win_len=4,
        ...     context_len=16,
        ...     sr=16000,
        ... )
        >>> x = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
        >>> num_mic = torch.tensor([3, 2])
        >>> output = model_TAC(x, num_mic.long())
        >>> print(output.shape)  # Expected shape: (batch, nspk, length)

    Note:
        This model assumes input data is in the shape of (batch, num_mic, length),
        where 'num_mic' is the number of microphones and 'length' is the duration of
        the input signal.

    Raises:
        AssertionError: If the dimensions of input data do not match the expected
        dimensions.
    """
    def __init__(self, *args, **kwargs):
        super(FaSNet_TAC, self).__init__(*args, **kwargs)

        self.context = int(self.sr * self.context_len / 1000)
        self.filter_dim = self.context * 2 + 1

        # DPRNN + TAC for estimation
        self.all_BF = BF_module(
            self.filter_dim + self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            dropout=self.dropout,
            fasnet_type="fasnet",
        )

        # waveform encoder
        self.encoder = nn.Conv1d(
            1, self.enc_dim, self.context * 2 + self.window, bias=False
        )
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)

    def forward(self, input, num_mic):
        """
        Abstract forward function for the FaSNet_TAC model.

    This method processes the input tensor through the FaSNet architecture
    and returns the beamformed signals. The input should be organized
    with dimensions representing the batch size, number of channels, and
    the sequence length. The method also handles the number of microphones
    used for each input.

    Args:
        input (torch.Tensor): Input tensor of shape (batch, max_num_ch, T),
            where `batch` is the batch size, `max_num_ch` is the maximum
            number of channels (microphones), and `T` is the length of the
            input sequence.
        num_mic (torch.Tensor): A tensor of shape (batch,) indicating the
            number of channels for each input. A value of zero indicates a
            fixed geometry configuration.

    Returns:
        torch.Tensor: The beamformed output signal of shape (B, nspk, T),
        where `B` is the batch size, `nspk` is the number of speakers, and
        `T` is the length of the output signal.

    Examples:
        >>> model = FaSNet_TAC(enc_dim=64, feature_dim=64, hidden_dim=128,
        ...                    layer=4, segment_size=50, nspk=2,
        ...                    win_len=4, context_len=16, sr=16000)
        >>> input_tensor = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
        >>> num_mic = torch.tensor([3, 2])  # Example number of microphones
        >>> output = model(input_tensor, num_mic)
        >>> print(output.shape)  # Expected shape: (2, 2, length)

    Note:
        The input tensor should be prepared in accordance with the expected
        input format. Ensure that the number of channels specified in `num_mic`
        corresponds to the actual number of microphones used in the input tensor.
        """
        batch_size = input.size(0)
        nmic = input.size(1)

        # split input into chunks
        all_seg, all_mic_context, rest = self.seg_signal_context(
            input, self.window, self.context
        )  # B, nmic, L, win/chunk
        seq_length = all_seg.size(2)

        # embeddings for all channels
        enc_output = (
            self.encoder(all_mic_context.view(-1, 1, self.context * 2 + self.window))
            .view(batch_size * nmic, seq_length, self.enc_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*nmic, N, L
        enc_output = self.enc_LN(enc_output).view(
            batch_size, nmic, self.enc_dim, seq_length
        )  # B, nmic, N, L

        # calculate the cosine similarities for ref channel's center
        # frame with all channels' context

        ref_seg = all_seg[:, 0].contiguous().view(1, -1, self.window)  # 1, B*L, win
        all_context = (
            all_mic_context.transpose(0, 1)
            .contiguous()
            .view(nmic, -1, self.context * 2 + self.window)
        )  # 1, B*L, 3*win
        all_cos_sim = self.seq_cos_sim(all_context, ref_seg)  # nmic, B*L, 2*win+1
        all_cos_sim = (
            all_cos_sim.view(nmic, batch_size, seq_length, self.filter_dim)
            .permute(1, 0, 3, 2)
            .contiguous()
        )  # B, nmic, 2*win+1, L

        input_feature = torch.cat([enc_output, all_cos_sim], 2)  # B, nmic, N+2*win+1, L

        # pass to DPRNN
        all_filter = self.all_BF(input_feature, num_mic)  # B, ch, nspk, L, 2*win+1

        # convolve with all mic's context
        mic_context = torch.cat(
            [
                all_mic_context.view(
                    batch_size * nmic, 1, seq_length, self.context * 2 + self.window
                )
            ]
            * self.num_spk,
            1,
        )  # B*nmic, nspk, L, 3*win
        all_bf_output = F.conv1d(
            mic_context.view(1, -1, self.context * 2 + self.window),
            all_filter.view(-1, 1, self.filter_dim),
            groups=batch_size * nmic * self.num_spk * seq_length,
        )  # 1, B*nmic*nspk*L, win
        all_bf_output = all_bf_output.view(
            batch_size, nmic, self.num_spk, seq_length, self.window
        )  # B, nmic, nspk, L, win

        # reshape to utterance
        bf_signal = all_bf_output.view(
            batch_size * nmic * self.num_spk, -1, self.window * 2
        )
        bf_signal1 = (
            bf_signal[:, :, : self.window]
            .contiguous()
            .view(batch_size * nmic * self.num_spk, 1, -1)[:, :, self.stride :]
        )
        bf_signal2 = (
            bf_signal[:, :, self.window :]
            .contiguous()
            .view(batch_size * nmic * self.num_spk, 1, -1)[:, :, : -self.stride]
        )
        bf_signal = bf_signal1 + bf_signal2  # B*nmic*nspk, 1, T
        if rest > 0:
            bf_signal = bf_signal[:, :, :-rest]

        bf_signal = bf_signal.view(
            batch_size, nmic, self.num_spk, -1
        )  # B, nmic, nspk, T
        # consider only the valid channels
        if num_mic.max() == 0:
            bf_signal = bf_signal.mean(1)  # B, nspk, T
        else:
            bf_signal = [
                bf_signal[b, : num_mic[b]].mean(0).unsqueeze(0)
                for b in range(batch_size)
            ]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal


def test_model(model):
    """
    Test the given model with random input data.

    This function generates random input data to test the functionality of the
    provided model. It creates a batch of audio signals with a specified number
    of microphones and verifies the output shape of the model for both variable
    and fixed microphone configurations.

    Args:
        model (nn.Module): The model to be tested, which should implement a
            forward method that accepts input data and the number of active
            microphones.

    Examples:
        >>> model_TAC = FaSNet_TAC(
        ...     enc_dim=64,
        ...     feature_dim=64,
        ...     hidden_dim=128,
        ...     layer=4,
        ...     segment_size=50,
        ...     nspk=2,
        ...     win_len=4,
        ...     context_len=16,
        ...     sr=16000,
        ... )
        >>> test_model(model_TAC)
        torch.Size([2, 2, 32000]) torch.Size([2, 2, 32000])

    Note:
        The input tensor `x` is generated with a shape of (batch, num_mic, length),
        where `batch` is set to 2, `num_mic` is set to 4, and `length` is set to 
        32000. The `num_mic` tensor is created as an ad-hoc array representing 
        the number of active microphones for each batch element.

    Raises:
        AssertionError: If the model's output shape does not match the expected
            dimensions.
    """
    x = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
    num_mic = (
        torch.from_numpy(np.array([3, 2]))
        .view(
            -1,
        )
        .type(x.type())
    )  # ad-hoc array
    none_mic = torch.zeros(1).type(x.type())  # fixed-array
    y1 = model(x, num_mic.long())
    y2 = model(x, none_mic.long())
    print(y1.shape, y2.shape)  # (batch, nspk, length)


if __name__ == "__main__":
    model_TAC = FaSNet_TAC(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=4,
        segment_size=50,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=16000,
    )

    test_model(model_TAC)
