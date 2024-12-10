import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers.dpmulcat import DPMulCat
from espnet2.enh.layers.dprnn import merge_feature, split_feature
from espnet2.enh.separator.abs_separator import AbsSeparator


def overlap_and_add(signal, frame_step):
    """
    Reconstructs a signal from a framed representation.

    This function adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length.

    Args:
        signal: A Tensor of shape [..., frames, frame_length]. All dimensions may
            be unknown, and the rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or
            equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames
        of signal's inner-most two dimensions. The output_size is calculated as:
        output_size = (frames - 1) * frame_step + frame_length.

    Examples:
        >>> import torch
        >>> signal = torch.randn(2, 4, 8)  # 2 batches, 4 frames, 8 length
        >>> frame_step = 4
        >>> output = overlap_and_add(signal, frame_step)
        >>> output.shape
        torch.Size([2, 8])  # output size should be (4 - 1) * 4 + 8 = 8

    Note:
        This function is based on the implementation found in TensorFlow:
        https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    # gcd=Greatest Common Divisor
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.clone().detach().long().to(signal.device)
    # frame = signal.new_tensor(frame).clone().long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Encoder(nn.Module):
    """
    Encoder module for processing input signals.

    This module utilizes a 1D convolutional layer followed by a ReLU
    activation function to transform the input mixture signal into
    a feature representation.

    Attributes:
        conv (nn.Conv1d): A convolutional layer that applies a 1D
            convolution to the input signal.
        nonlinear (nn.ReLU): A ReLU activation function applied to the
            output of the convolutional layer.

    Args:
        enc_kernel_size (int): The size of the kernel used in the
            convolutional layer.
        enc_feat_dim (int): The dimension of the feature output
            from the encoder.

    Examples:
        >>> encoder = Encoder(enc_kernel_size=8, enc_feat_dim=128)
        >>> mixture = torch.randn(10, 160)  # Example batch of signals
        >>> output = encoder(mixture)
        >>> output.shape
        torch.Size([10, 128, 80])  # Example output shape

    Note:
        The input mixture signal should have a shape of
        [batch_size, signal_length].
    """

    def __init__(self, enc_kernel_size: int, enc_feat_dim: int):
        super().__init__()
        # setting 50% overlap
        self.conv = nn.Conv1d(
            1,
            enc_feat_dim,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            bias=False,
        )
        self.nonlinear = nn.ReLU()

    def forward(self, mixture):
        """
        Performs the forward pass of the SVoiceSeparator model.

        This method processes the input tensor through the encoder,
        RNN model, and decoder to separate audio sources.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature of shape [B, T, N],
                where B is the batch size, T is the time dimension, and N is the
                number of frequency bins.
            ilens (torch.Tensor): A tensor of shape [Batch] representing the input
                lengths for each instance in the batch.
            additional (Dict or None): Optional dictionary containing other data
                included in the model. NOTE: This parameter is not used in this model.

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): A list of tensors
                with shape [(B, T, N), ...] representing the separated sources.
            ilens (torch.Tensor): A tensor of shape (B,) representing the lengths
                of the input sequences.
            others (OrderedDict): A dictionary containing additional predicted data,
                such as masks for each speaker:
                - 'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                - 'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                - ...
                - 'mask_spkn': torch.Tensor(Batch, Frames, Freq).

        Examples:
            >>> model = SVoiceSeparator(input_dim=256, enc_dim=128, kernel_size=8)
            >>> input_tensor = torch.randn(2, 100, 256)  # Batch of 2
            >>> ilens = torch.tensor([100, 90])  # Input lengths
            >>> outputs, lengths, masks = model(input_tensor, ilens)

        Note:
            The time dimension might change due to convolution operations.
        """
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = self.nonlinear(self.conv(mixture))
        return mixture_w


class Decoder(nn.Module):
    """
    Decoder module for reconstructing audio signals from estimated sources.

    The Decoder takes the estimated source signals and applies an average pooling
    operation followed by an overlap-and-add procedure to reconstruct the time-domain
    signal from its framed representation.

    Args:
        kernel_size (int): The size of the kernel used for the average pooling operation.

    Returns:
        torch.Tensor: The reconstructed time-domain signal from the estimated sources.

    Examples:
        >>> decoder = Decoder(kernel_size=8)
        >>> est_source = torch.rand(1, 2, 10, 8)  # Example estimated source tensor
        >>> reconstructed_signal = decoder(est_source)
        >>> print(reconstructed_signal.shape)
        torch.Size([1, output_length])  # The output shape will depend on the input size.

    Note:
        The overlap-and-add method requires that the kernel size is greater than zero.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, est_source):
        """
            Perform a forward pass through the SVoice separator model.

        This method processes the input tensor through the encoder, applies
        a dual-path RNN model for separation, and decodes the output to produce
        separated speech signals for multiple speakers.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature tensor
                of shape [B, T, N], where B is the batch size, T is the
                number of time frames, and N is the number of frequency bins.
            ilens (torch.Tensor): Input lengths tensor of shape [Batch],
                indicating the length of each input sequence in the batch.
            additional (Dict or None): A dictionary containing other data
                included in the model. This argument is not used in this model.

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): A list of
                tensors, each representing the separated output for a speaker,
                in the shape [(B, T, N), ...].
            ilens (torch.Tensor): A tensor of shape (B,) containing the
                lengths of the input sequences.
            others (OrderedDict): An ordered dictionary containing predicted
                data, such as masks for each speaker:
                - 'mask_spk1': torch.Tensor(Batch, Frames, Freq)
                - 'mask_spk2': torch.Tensor(Batch, Frames, Freq)
                - ...
                - 'mask_spkn': torch.Tensor(Batch, Frames, Freq)

        Examples:
            >>> model = SVoiceSeparator(input_dim=128, enc_dim=128, kernel_size=8)
            >>> input_tensor = torch.randn(2, 100, 128)  # Example input
            >>> ilens = torch.tensor([100, 100])  # Example input lengths
            >>> outputs, lengths, masks = model(input_tensor, ilens)

        Note:
            The time dimension of the input may be altered due to convolution
            operations. Ensure that the output is padded back to the original
            length for proper alignment.
        """
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.kernel_size))(est_source)
        est_source = overlap_and_add(est_source, self.kernel_size // 2)

        return est_source


class SVoiceSeparator(AbsSeparator):
    """
    SVoice model for speech separation.

    This model implements the SVoice architecture for separating multiple
    speakers from a mixed audio input. It utilizes an encoder-decoder
    structure combined with a dual-path RNN model to effectively process
    audio signals with an unknown number of speakers.

    Reference:
        Voice Separation with an Unknown Number of Multiple Speakers;
        E. Nachmani et al., 2020;
        https://arxiv.org/abs/2003.01531

    Args:
        input_dim (int): Dimension of the input features.
        enc_dim (int): Dimension of the encoder module's output. (Default: 128)
        kernel_size (int): The kernel size of Conv1D layer in both encoder
            and decoder modules. (Default: 8)
        hidden_size (int): Dimension of the hidden state in RNN layers.
            (Default: 128)
        num_spk (int): The number of speakers in the output. (Default: 2)
        num_layers (int): Number of stacked MulCat blocks. (Default: 4)
        segment_size (int): Dual-path segment size. (Default: 20)
        bidirectional (bool): Whether the RNN layers are bidirectional.
            (Default: True)
        input_normalize (bool): Whether to apply GroupNorm on the input Tensor.
            (Default: False)

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]: A tuple containing:
            - masked (List[Union(torch.Tensor, ComplexTensor)]): A list of
              tensors representing the separated sources for each speaker,
              with shape [(B, T, N), ...].
            - ilens (torch.Tensor): A tensor representing the lengths of the
              input sequences, with shape (B,).
            - others (OrderedDict): A dictionary containing additional
              predicted data, such as masks for each speaker:
              {
                  'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                  'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                  ...
                  'mask_spkn': torch.Tensor(Batch, Frames, Freq),
              }

    Examples:
        >>> separator = SVoiceSeparator(input_dim=256, enc_dim=128, kernel_size=8)
        >>> input_tensor = torch.randn(10, 100, 256)  # Batch of 10, 100 time steps
        >>> input_lengths = torch.tensor([100] * 10)  # All sequences have length 100
        >>> outputs, lengths, masks = separator(input_tensor, input_lengths)

    Note:
        The `additional` argument is not used in this model but is included
        for compatibility with the general interface of the separator class.
    """

    def __init__(
        self,
        input_dim: int,
        enc_dim: int,
        kernel_size: int,
        hidden_size: int,
        num_spk: int = 2,
        num_layers: int = 4,
        segment_size: int = 20,
        bidirectional: bool = True,
        input_normalize: bool = False,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.enc_dim = enc_dim
        self.segment_size = segment_size
        # model sub-networks
        self.encoder = Encoder(kernel_size, enc_dim)
        self.decoder = Decoder(kernel_size)
        self.rnn_model = DPMulCat(
            input_size=enc_dim,
            hidden_size=hidden_size,
            output_size=enc_dim,
            num_spk=num_spk,
            num_layers=num_layers,
            bidirectional=bidirectional,
            input_normalize=input_normalize,
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """
            SVoice model for speech separation.

        This class implements the SVoice model, which is designed for separating
        speech from multiple speakers in a given audio input. It utilizes an
        encoder-decoder architecture with a recurrent neural network (RNN) for
        effective separation.

        Reference:
            Voice Separation with an Unknown Number of Multiple Speakers;
            E. Nachmani et al., 2020;
            https://arxiv.org/abs/2003.01531

        Attributes:
            enc_dim (int): Dimension of the encoder module's output.
            kernel_size (int): The kernel size of Conv1D layer in both encoder
                and decoder modules.
            hidden_size (int): Dimension of the hidden state in RNN layers.
            num_spk (int): The number of speakers in the output.
            num_layers (int): Number of stacked MulCat blocks.
            segment_size (int): Dual-path segment size.
            bidirectional (bool): Whether the RNN layers are bidirectional.
            input_normalize (bool): Whether to apply GroupNorm on the input Tensor.

        Args:
            input_dim (int): Dimension of the input feature.
            enc_dim (int): Dimension of the encoder module's output.
            kernel_size (int): The kernel size of Conv1D layer in both encoder
                and decoder modules.
            hidden_size (int): Dimension of the hidden state in RNN layers.
            num_spk (int, optional): The number of speakers in the output.
                (Default: 2)
            num_layers (int, optional): Number of stacked MulCat blocks.
                (Default: 4)
            segment_size (int, optional): Dual-path segment size.
                (Default: 20)
            bidirectional (bool, optional): Whether the RNN layers are
                bidirectional. (Default: True)
            input_normalize (bool, optional): Whether to apply GroupNorm on
                the input Tensor. (Default: False)

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
                - masked: List of tensors containing separated audio signals
                  for each speaker.
                - ilens: Tensor containing the lengths of the input sequences.
                - others: An OrderedDict containing any additional predicted data
                  such as masks for each speaker.

        Examples:
            >>> model = SVoiceSeparator(input_dim=512, enc_dim=128,
            ...                         kernel_size=8, hidden_size=128)
            >>> input_tensor = torch.randn(2, 100, 512)  # Batch of 2, 100 time steps
            >>> ilens = torch.tensor([100, 100])  # Lengths of each input
            >>> outputs, ilens, others = model(input_tensor, ilens)

        Note:
            The `additional` argument is not used in this model but is included
            for compatibility with other models.
        """
        # fix time dimension, might change due to convolution operations
        T_mix = input.size(-1)

        mixture_w = self.encoder(input)

        enc_segments, enc_rest = split_feature(mixture_w, self.segment_size)
        # separate
        output_all = self.rnn_model(enc_segments)

        # generate wav after each RNN block and optimize the loss
        outputs = []
        for ii in range(len(output_all)):
            output_ii = merge_feature(output_all[ii], enc_rest)
            output_ii = output_ii.view(
                input.shape[0], self._num_spk, self.enc_dim, mixture_w.shape[2]
            )
            output_ii = self.decoder(output_ii)
            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            output_ii = list(output_ii.unbind(dim=1))
            if self.training:
                outputs.append(output_ii)
            else:
                outputs = output_ii

        others = {}
        return outputs, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
