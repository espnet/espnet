# The implementation of DPRNN in
# Luo. et al. "Dual-path rnn: efficient long sequence modeling
# for time-domain single-channel speech separation."
#
# The code is based on:
# https://github.com/yluo42/TAC/blob/master/utility/models.py
# Licensed under CC BY-NC-SA 3.0 US.
#


import torch
import torch.nn as nn
from torch.autograd import Variable

EPS = torch.finfo(torch.get_default_dtype()).eps


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    This class implements a single RNN layer that can be configured to use
    either RNN, LSTM, or GRU types. It includes a linear projection layer
    and dropout functionality.

    Attributes:
        rnn_type (str): The type of RNN ('RNN', 'LSTM', or 'GRU').
        input_size (int): The dimension of the input features.
        hidden_size (int): The dimension of the hidden state.
        num_direction (int): The number of directions for the RNN (1 for
            unidirectional, 2 for bidirectional).
        rnn (nn.Module): The RNN layer instance.
        dropout (nn.Dropout): The dropout layer.
        proj (nn.Linear): The linear projection layer.

    Args:
        rnn_type (str): Type of RNN to use. Must be 'RNN', 'LSTM', or 'GRU'.
        input_size (int): Dimension of the input feature. The input should
            have shape (batch, seq_len, input_size).
        hidden_size (int): Dimension of the hidden state.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is False.

    Raises:
        AssertionError: If rnn_type is not one of 'RNN', 'LSTM', or 'GRU'.

    Examples:
        >>> rnn_layer = SingleRNN('LSTM', input_size=128, hidden_size=64)
        >>> input_tensor = torch.randn(32, 10, 128)  # (batch, seq_len, input_size)
        >>> output, state = rnn_layer(input_tensor)
        >>> output.shape
        torch.Size([32, 10, 128])  # output shape is (batch, seq_len, input_size)
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super().__init__()

        rnn_type = rnn_type.upper()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input, state=None):
        """
            Perform a forward pass through the SingleRNN layer.

        This method takes the input tensor and an optional hidden state,
        processes the input through the RNN layer, applies dropout, and
        projects the output back to the input feature dimension.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            state (torch.Tensor, optional): The initial hidden state of the RNN.
                                             If not provided, it defaults to None.

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): The output tensor after processing, with
                  shape (batch, seq_len, input_size).
                - state (torch.Tensor): The hidden state after processing, with
                  shape (num_layers * num_directions, batch, hidden_size).

        Examples:
            >>> rnn = SingleRNN(rnn_type='LSTM', input_size=10, hidden_size=20)
            >>> input_tensor = torch.randn(5, 15, 10)  # (batch_size, seq_len, input_size)
            >>> output, hidden_state = rnn(input_tensor)

        Note:
            The input tensor should have shape (batch, seq_len, input_size).
            The hidden state should match the dimensions of the RNN layer.
        """
        # input shape: batch, seq, dim
        # input = input.to(device)
        output = input
        rnn_output, state = self.rnn(output, state)
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output, state


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep dual-path RNN for efficient long sequence modeling.

    This module implements a dual-path RNN as proposed in Luo et al.
    "Dual-path RNN: efficient long sequence modeling for time-domain
    single-channel speech separation." The dual-path RNN applies RNN
    layers in both row and column directions to effectively model
    long sequences while maintaining efficiency.

    Args:
        rnn_type (str): Select from 'RNN', 'LSTM', and 'GRU'.
        input_size (int): Dimension of the input feature. The input
                          should have shape (batch, seq_len, input_size).
        hidden_size (int): Dimension of the hidden state.
        output_size (int): Dimension of the output size.
        dropout (float): Dropout ratio. Default is 0.
        num_layers (int): Number of stacked RNN layers. Default is 1.
        bidirectional (bool): Whether the RNN layers are bidirectional.
                             Default is True.

    Examples:
        >>> dprnn = DPRNN('LSTM', input_size=256, hidden_size=128,
                          output_size=256, num_layers=2)
        >>> input_tensor = torch.randn(32, 10, 256)  # (batch_size, seq_len, input_size)
        >>> output = dprnn(input_tensor)
        >>> print(output.shape)  # Should be (32, 256, 10, 1)

    Returns:
        torch.Tensor: The output tensor with shape
                      (batch_size, output_size, dim1, dim2).

    Note:
        The dual-path RNN consists of row RNNs and column RNNs that are
        applied in sequence. The output is then processed through a
        linear layer for final predictions.

    Raises:
        AssertionError: If rnn_type is not one of 'RNN', 'LSTM', or 'GRU'.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dropout=0,
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(
                    rnn_type, input_size, hidden_size, dropout, bidirectional=True
                )
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(
                    rnn_type,
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN.
            # For causal setting change it to causal normalization accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        """
            Perform a forward pass through the RNN layer.

        This method processes the input tensor through the RNN layer, applies
        dropout, and performs a linear projection to transform the RNN output
        back to the input feature space.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
                It should have dimensions representing batch size, sequence
                length, and input feature dimension.
            state (torch.Tensor, optional): The initial hidden state for the RNN.
                If not provided, the RNN will initialize its hidden state
                internally.

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): The output tensor after processing through
                  the RNN layer, of shape (batch, seq_len, dim).
                - state (torch.Tensor): The final hidden state of the RNN.

        Examples:
            >>> rnn = SingleRNN('LSTM', input_size=10, hidden_size=20)
            >>> input_tensor = torch.randn(5, 15, 10)  # batch_size=5, seq_len=15
            >>> output, state = rnn(input_tensor)

        Note:
            The input tensor must be of shape (batch, seq_len, input_size)
            and the output tensor will have the same shape as the input tensor.
        """
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            row_output, _ = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = (
                row_output.view(batch_size, dim2, dim1, -1)
                .permute(0, 3, 2, 1)
                .contiguous()
            )  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            col_output, _ = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = (
                col_output.view(batch_size, dim1, dim2, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output)  # B, output_size, dim1, dim2

        return output


# dual-path RNN with transform-average-concatenate (TAC)
class DPRNN_TAC(nn.Module):
    """
    Deep duaL-path RNN with TAC applied to each layer/block.

    This class implements a deep dual-path RNN architecture with
    Transform-Average-Concatenate (TAC) applied to each layer/block.
    It is designed for efficient processing of 3D input data and
    leverages the capabilities of RNNs for time-domain single-channel
    speech separation.

    Attributes:
        input_size (int): Dimension of the input feature.
        output_size (int): Dimension of the output size.
        hidden_size (int): Dimension of the hidden state.
        row_rnn (nn.ModuleList): List of row RNNs for intra-segment processing.
        col_rnn (nn.ModuleList): List of column RNNs for inter-segment processing.
        ch_transform (nn.ModuleList): List of transformation layers for channels.
        ch_average (nn.ModuleList): List of average pooling layers for channels.
        ch_concat (nn.ModuleList): List of concatenation layers for channels.
        row_norm (nn.ModuleList): List of normalization layers for row outputs.
        col_norm (nn.ModuleList): List of normalization layers for column outputs.
        ch_norm (nn.ModuleList): List of normalization layers for channel outputs.
        output (nn.Sequential): Output layer that processes the final output.

    Args:
        rnn_type (str): Type of RNN to use. Must be one of 'RNN', 'LSTM', or 'GRU'.
        input_size (int): Dimension of the input feature. The input should
                          have shape (batch, seq_len, input_size).
        hidden_size (int): Dimension of the hidden state.
        output_size (int): Dimension of the output size.
        dropout (float): Dropout ratio. Default is 0.
        num_layers (int): Number of stacked RNN layers. Default is 1.
        bidirectional (bool): Whether the RNN layers are bidirectional.
                             Default is False.

    Examples:
        >>> model = DPRNN_TAC(rnn_type='LSTM', input_size=64, hidden_size=128,
        ...                   output_size=64, dropout=0.1, num_layers=2)
        >>> input_tensor = torch.randn(32, 4, 16, 64)  # (batch, ch, N, dim1, dim2)
        >>> num_mic = torch.tensor([2] * 32)  # Assume all inputs have 2 microphones
        >>> output = model(input_tensor, num_mic)

    Note:
        The model supports both fixed geometry arrays and variable geometry arrays
        based on the `num_mic` parameter.

    Raises:
        AssertionError: If `rnn_type` is not one of the supported types ('RNN', 'LSTM', 'GRU').
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dropout=0,
        num_layers=1,
        bidirectional=True,
    ):
        super(DPRNN_TAC, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # DPRNN + TAC for 3D input (ch, N, T)
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.ch_transform = nn.ModuleList([])
        self.ch_average = nn.ModuleList([])
        self.ch_concat = nn.ModuleList([])

        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.ch_norm = nn.ModuleList([])

        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(
                    rnn_type, input_size, hidden_size, dropout, bidirectional=True
                )
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(
                    rnn_type,
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            self.ch_transform.append(
                nn.Sequential(nn.Linear(input_size, hidden_size * 3), nn.PReLU())
            )
            self.ch_average.append(
                nn.Sequential(nn.Linear(hidden_size * 3, hidden_size * 3), nn.PReLU())
            )
            self.ch_concat.append(
                nn.Sequential(nn.Linear(hidden_size * 6, input_size), nn.PReLU())
            )

            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for
            # inter-chunk RNN and TAC modules.
            # For causal setting change them to causal normalization
            # techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.ch_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input, num_mic):
        """
            Forward pass for the DPRNN_TAC model.

        This method processes the input through the dual-path RNN with TAC
        applied to each layer/block. The model first applies RNN on the
        'dim1' dimension, followed by 'dim2', and finally across channels.

        Args:
            input (torch.Tensor): Input tensor of shape
                (batch, ch, N, dim1, dim2), where 'ch' is the number of
                channels, 'N' is the sequence length, and 'dim1', 'dim2'
                are the dimensions of the input features.
            num_mic (torch.Tensor): A tensor of shape (batch,) indicating
                the number of microphones used for each batch item.

        Returns:
            torch.Tensor: The output tensor after processing, of shape
                (B, ch, N, dim1, dim2), where 'B' is the batch size.

        Examples:
            >>> model = DPRNN_TAC('LSTM', input_size=64, hidden_size=128,
            ...                   output_size=64)
            >>> input_tensor = torch.randn(10, 4, 20, 32, 32)  # Batch of 10
            >>> num_mic = torch.tensor([2, 2, 1, 0, 2, 1, 2, 0, 1, 2])
            >>> output = model(input_tensor, num_mic)
            >>> output.shape
            torch.Size([10, 4, 20, 32, 32])
        """
        # input shape: batch, ch, N, dim1, dim2
        # num_mic shape: batch,
        # apply RNN on dim1 first, then dim2, then ch

        batch_size, ch, N, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            # intra-segment RNN
            output = output.view(batch_size * ch, N, dim1, dim2)
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * ch * dim2, dim1, -1)
            )  # B*ch*dim2, dim1, N
            row_output, _ = self.row_rnn[i](row_input)  # B*ch*dim2, dim1, N
            row_output = (
                row_output.view(batch_size * ch, dim2, dim1, -1)
                .permute(0, 3, 2, 1)
                .contiguous()
            )  # B*ch, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output  # B*ch, N, dim1, dim2

            # inter-segment RNN
            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * ch * dim1, dim2, -1)
            )  # B*ch*dim1, dim2, N
            col_output, _ = self.col_rnn[i](col_input)  # B*dim1, dim2, N
            col_output = (
                col_output.view(batch_size * ch, dim1, dim2, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # B*ch, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output  # B*ch, N, dim1, dim2

            # TAC for cross-channel communication
            ch_input = output.view(input.shape)  # B, ch, N, dim1, dim2
            ch_input = (
                ch_input.permute(0, 3, 4, 1, 2).contiguous().view(-1, N)
            )  # B*dim1*dim2*ch, N
            ch_output = self.ch_transform[i](ch_input).view(
                batch_size, dim1 * dim2, ch, -1
            )  # B, dim1*dim2, ch, H
            # mean pooling across channels
            if num_mic.max() == 0:
                # fixed geometry array
                ch_mean = ch_output.mean(2).view(
                    batch_size * dim1 * dim2, -1
                )  # B*dim1*dim2, H
            else:
                # only consider valid channels
                ch_mean = [
                    ch_output[b, :, : num_mic[b]].mean(1).unsqueeze(0)
                    for b in range(batch_size)
                ]  # 1, dim1*dim2, H
                ch_mean = torch.cat(ch_mean, 0).view(
                    batch_size * dim1 * dim2, -1
                )  # B*dim1*dim2, H
            ch_output = ch_output.view(
                batch_size * dim1 * dim2, ch, -1
            )  # B*dim1*dim2, ch, H
            ch_mean = (
                self.ch_average[i](ch_mean)
                .unsqueeze(1)
                .expand_as(ch_output)
                .contiguous()
            )  # B*dim1*dim2, ch, H
            ch_output = torch.cat([ch_output, ch_mean], 2)  # B*dim1*dim2, ch, 2H
            ch_output = self.ch_concat[i](
                ch_output.view(-1, ch_output.shape[-1])
            )  # B*dim1*dim2*ch, N
            ch_output = (
                ch_output.view(batch_size, dim1, dim2, ch, -1)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )  # B, ch, N, dim1, dim2
            ch_output = self.ch_norm[i](
                ch_output.view(batch_size * ch, N, dim1, dim2)
            )  # B*ch, N, dim1, dim2
            output = output + ch_output

        output = self.output(output)  # B*ch, N, dim1, dim2

        return output


def _pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, segment_size):
    """
    Split the input features into chunks of specified segment size.

    This function takes a tensor of features and splits it into overlapping
    segments of a given size. It also handles padding to ensure that the
    segments are of the correct size and can be processed without losing
    information from the input tensor.

    Args:
        input (torch.Tensor): The input tensor of shape (B, N, T), where
            B is the batch size, N is the number of features, and T is the
            sequence length.
        segment_size (int): The size of each segment to split the input into.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - A tensor of shape (B, N, K, segment_size), where K is the
              number of segments created from the input.
            - An integer representing the number of elements that were
              padded at the end of the input.

    Examples:
        >>> input_tensor = torch.randn(2, 3, 10)  # (B=2, N=3, T=10)
        >>> segments, rest = split_feature(input_tensor, segment_size=4)
        >>> segments.shape
        torch.Size([2, 3, 6, 4])  # Example output shape with K=6 segments

    Note:
        The function uses zero-padding to ensure that the input length is
        a multiple of the segment size before splitting.
    """
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = _pad_segment(input, segment_size)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = (
        input[:, :, :-segment_stride]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments2 = (
        input[:, :, segment_stride:]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments = (
        torch.cat([segments1, segments2], 3)
        .view(batch_size, dim, -1, segment_size)
        .transpose(2, 3)
    )

    return segments.contiguous(), rest


def merge_feature(input, rest):
    """
    Merge the splitted features into full utterance.

    This function takes the split features and reconstructs the original
    full utterance. It combines the segments produced by the `split_feature`
    function, accounting for any remaining elements that were padded during
    the segmentation process.

    Args:
        input (torch.Tensor): The input features with shape
            (B, N, L, K), where B is the batch size, N is the number of
            features, L is the number of segments, and K is the segment size.
        rest (int): The number of elements that were padded and should
            be removed from the output.

    Returns:
        torch.Tensor: The reconstructed features with shape (B, N, T),
        where T is the length of the original sequence after removing
        the padded elements.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)  # Example input tensor
        >>> rest = 1  # Example rest value
        >>> output = merge_feature(input, rest)
        >>> print(output.shape)  # Should output: torch.Size([2, 3, 19])

    Note:
        The `rest` parameter should match the padding applied during the
        splitting process to ensure correct reconstruction of the original
        sequence length.
    """
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
    )  # B, N, K, L

    input1 = (
        input[:, :, :, :segment_size]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, segment_stride:]
    )
    input2 = (
        input[:, :, :, segment_size:]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, :-segment_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T
