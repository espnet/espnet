# The implementation of DPTNet proposed in
# J. Chen, Q. Mao, and D. Liu, “Dual-path transformer network:
# Direct context-aware modeling for end-to-end monaural speech
# separation,” in Proc. ISCA Interspeech, 2020, pp. 2642–2646.
#
# Ported from https://github.com/ujscjj/DPTNet

import torch.nn as nn

from espnet2.enh.layers.tcn import choose_norm
from espnet.nets.pytorch_backend.nets_utils import get_activation


class ImprovedTransformerLayer(nn.Module):
    """
    Container module of the (improved) Transformer proposed in [1].

    This class implements the Improved Transformer Layer as part of the Dual-Path 
    Transformer Network (DPTNet) architecture. It incorporates a multi-head self-attention 
    mechanism followed by a feed-forward network, and can utilize various RNN types for 
    processing the input features. This layer is designed for applications such as 
    end-to-end monaural speech separation.

    Reference:
        Chen, J., Mao, Q., & Liu, D. (2020). Dual-path transformer network: Direct 
        context-aware modeling for end-to-end monaural speech separation. In Proc. 
        ISCA Interspeech (pp. 2642–2646).

    Attributes:
        rnn_type (str): Type of RNN used ('RNN', 'LSTM', or 'GRU').
        att_heads (int): Number of attention heads.
        self_attn (nn.MultiheadAttention): Multi-head self-attention layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        norm_attn: Normalization layer for attention output.
        rnn (nn.Module): RNN layer based on specified rnn_type.
        feed_forward (nn.Sequential): Feed-forward network following the RNN.
        norm_ff: Normalization layer for feed-forward output.

    Args:
        rnn_type (str): Select from 'RNN', 'LSTM', and 'GRU'.
        input_size (int): Dimension of the input feature.
        att_heads (int): Number of attention heads.
        hidden_size (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        activation (str): Activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN 
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.

    Examples:
        >>> layer = ImprovedTransformerLayer(
        ...     rnn_type='LSTM',
        ...     input_size=256,
        ...     att_heads=4,
        ...     hidden_size=128,
        ...     dropout=0.1,
        ...     activation='relu'
        ... )
        >>> input_tensor = torch.randn(10, 20, 256)  # (batch, seq_len, input_size)
        >>> output_tensor = layer(input_tensor)

    Raises:
        AssertionError: If rnn_type is not one of 'RNN', 'LSTM', or 'GRU'.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        att_heads,
        hidden_size,
        dropout=0.0,
        activation="relu",
        bidirectional=True,
        norm="gLN",
    ):
        super().__init__()

        rnn_type = rnn_type.upper()
        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"
        self.rnn_type = rnn_type

        self.att_heads = att_heads
        self.self_attn = nn.MultiheadAttention(input_size, att_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_attn = choose_norm(norm, input_size)

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        hdim = 2 * hidden_size if bidirectional else hidden_size
        if activation.lower() == "linear":
            activation = nn.Identity()
        else:
            activation = get_activation(activation)
        self.feed_forward = nn.Sequential(
            activation, nn.Dropout(p=dropout), nn.Linear(hdim, input_size)
        )

        self.norm_ff = choose_norm(norm, input_size)

    def forward(self, x, attn_mask=None):
        """
        Forward pass through the Improved Transformer Layer.

    This method takes the input tensor `x`, applies self-attention,
    a feed-forward neural network, and normalization, returning the 
    transformed output.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, seq, input_size).
        attn_mask (torch.Tensor, optional): Attention mask to prevent
            attention to certain positions. Default is None.

    Returns:
        torch.Tensor: Output tensor of the same shape as input `x` 
        after applying the transformer layer.

    Examples:
        >>> layer = ImprovedTransformerLayer('LSTM', 128, 4, 64)
        >>> input_tensor = torch.randn(32, 10, 128)  # (batch, seq, input_size)
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)  # Should output: torch.Size([32, 10, 128])

    Note:
        The input tensor `x` should have dimensions corresponding to 
        (batch size, sequence length, input size).
        """
        # (batch, seq, input_size) -> (seq, batch, input_size)
        src = x.permute(1, 0, 2)
        # (seq, batch, input_size) -> (batch, seq, input_size)
        out = self.self_attn(src, src, src, attn_mask=attn_mask)[0].permute(1, 0, 2)
        out = self.dropout(out) + x
        # ... -> (batch, input_size, seq) -> ...
        out = self.norm_attn(out.transpose(-1, -2)).transpose(-1, -2)

        out2 = self.feed_forward(self.rnn(out)[0])
        out2 = self.dropout(out2) + out
        return self.norm_ff(out2.transpose(-1, -2)).transpose(-1, -2)


class DPTNet(nn.Module):
    """
    Dual-path transformer network.

This implementation of DPTNet is based on the work by J. Chen, Q. Mao, and 
D. Liu, “Dual-path transformer network: Direct context-aware modeling for 
end-to-end monaural speech separation,” presented at ISCA Interspeech, 
2020. It utilizes an improved transformer layer to process input data 
through a dual-path approach.

Attributes:
    input_size (int): Dimension of the input feature.
    hidden_size (int): Dimension of the hidden state.
    output_size (int): Dimension of the output size.
    row_transformer (nn.ModuleList): List of transformer layers for row 
        processing.
    col_transformer (nn.ModuleList): List of transformer layers for column 
        processing.
    output (nn.Sequential): Final output layer consisting of PReLU and 
        Conv2d.

Args:
    rnn_type (str): Select from 'RNN', 'LSTM', and 'GRU'.
    input_size (int): Dimension of the input feature. Input size must be a 
        multiple of `att_heads`.
    hidden_size (int): Dimension of the hidden state.
    output_size (int): Dimension of the output size.
    att_heads (int): Number of attention heads.
    dropout (float): Dropout ratio. Default is 0.
    activation (str): Activation function applied at the output of RNN.
    num_layers (int): Number of stacked RNN layers. Default is 1.
    bidirectional (bool): Whether the RNN layers are bidirectional. Default 
        is True.
    norm_type (str): Type of normalization to use after each inter- or 
        intra-chunk Transformer block.

Examples:
    >>> model = DPTNet(
    ...     rnn_type='LSTM',
    ...     input_size=256,
    ...     hidden_size=128,
    ...     output_size=256,
    ...     att_heads=4,
    ...     dropout=0.1,
    ...     activation='relu',
    ...     num_layers=2,
    ...     bidirectional=True,
    ...     norm_type='gLN'
    ... )
    >>> input_tensor = torch.randn(8, 10, 256, 5)  # Batch of 8
    >>> output_tensor = model(input_tensor)
    >>> print(output_tensor.shape)
    torch.Size([8, 256, 10, 5])  # Output shape

Note:
    The input tensor must be of shape (batch, N, dim1, dim2).

Raises:
    AssertionError: If the provided rnn_type is not one of 'RNN', 'LSTM', 
        or 'GRU'.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        num_layers=1,
        bidirectional=True,
        norm_type="gLN",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        for i in range(num_layers):
            self.row_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=True,
                    norm=norm_type,
                )
            )  # intra-segment RNN is always noncausal
            self.col_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
            )

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        """
        Perform the forward pass of the DPTNet model.

    This method processes the input tensor through the dual-path transformer
    network. It first applies the transformer on the first dimension and then 
    on the second dimension, resulting in the output tensor.

    Args:
        input (torch.Tensor): Input tensor of shape (batch, N, dim1, dim2), 
            where `batch` is the batch size, `N` is the number of features, 
            `dim1` is the first dimension, and `dim2` is the second dimension.

    Returns:
        torch.Tensor: Output tensor of shape (batch, output_size, dim1, dim2),
            where `output_size` is the dimension of the output size.

    Examples:
        >>> model = DPTNet(rnn_type='LSTM', input_size=256, hidden_size=128, 
        ...                output_size=10)
        >>> input_tensor = torch.randn(32, 64, 256, 128)  # Example input
        >>> output_tensor = model(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([32, 10, 64, 128])

    Note:
        The input tensor must have dimensions that match the expected shape.
        The model applies the intra-chunk and inter-chunk processes in a loop
        for the specified number of layers.
        """
        # input shape: batch, N, dim1, dim2
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        output = input
        for i in range(len(self.row_transformer)):
            output = self.intra_chunk_process(output, i)
            output = self.inter_chunk_process(output, i)

        output = self.output(output)  # B, output_size, dim1, dim2

        return output

    def intra_chunk_process(self, x, layer_index):
        """
        Processes input tensors through the intra-chunk transformer layer.

    This method reshapes the input tensor for the specified layer index, 
    applies the intra-chunk transformer processing, and reshapes the output 
    back to its original dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, N, chunk_size, n_chunks).
        layer_index (int): Index of the transformer layer to be used for processing.

    Returns:
        torch.Tensor: Transformed tensor of shape (batch, N, chunk_size, n_chunks).

    Examples:
        >>> model = DPTNet('LSTM', 128, 64, 32)
        >>> input_tensor = torch.randn(10, 16, 32, 4)  # Example input
        >>> output_tensor = model.intra_chunk_process(input_tensor, 0)
        >>> output_tensor.shape
        torch.Size([10, 32, 32, 4])  # Example output shape after processing

    Note:
        The input tensor is expected to have four dimensions corresponding 
        to batch size, number of features, chunk size, and number of chunks.
        """
        batch, N, chunk_size, n_chunks = x.size()
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, N)
        x = self.row_transformer[layer_index](x)
        x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1)
        return x

    def inter_chunk_process(self, x, layer_index):
        """
        Process the output from the intra-chunk transformer layer and apply the 
    inter-chunk transformer layer.

    This method reshapes the input tensor to allow processing across chunks 
    using the column transformer defined in the DPTNet architecture.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, N, chunk_size, n_chunks),
            where `batch` is the batch size, `N` is the feature dimension,
            `chunk_size` is the size of each chunk, and `n_chunks` is the 
            number of chunks.
        layer_index (int): The index of the current layer being processed.

    Returns:
        torch.Tensor: Output tensor of shape (batch, N, chunk_size, n_chunks)
            after applying the column transformer.

    Examples:
        >>> import torch
        >>> model = DPTNet('LSTM', 128, 64, 32)
        >>> input_tensor = torch.randn(16, 128, 10, 5)  # batch_size=16
        >>> output_tensor = model.inter_chunk_process(input_tensor, 0)
        >>> output_tensor.shape
        torch.Size([16, 32, 10, 5])

    Note:
        The input tensor is expected to be in the format (batch, N, chunk_size, 
        n_chunks) prior to calling this method.
        """
        batch, N, chunk_size, n_chunks = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch * chunk_size, n_chunks, N)
        x = self.col_transformer[layer_index](x)
        x = x.reshape(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return x
