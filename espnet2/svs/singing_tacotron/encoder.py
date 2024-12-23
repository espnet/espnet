#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
# Copyright 2023 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Singing Tacotron encoder related modules."""

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def encoder_init(m):
    """
    Initialize encoder parameters.

    This function initializes the parameters of the encoder module, specifically
    for convolutional layers. It applies the Xavier uniform initialization to
    the weights of Conv1d layers to improve convergence during training.

    Args:
        m (torch.nn.Module): The neural network module to initialize.

    Note:
        This function is typically used with the `apply` method of a PyTorch
        module to initialize all its sub-modules.

    Examples:
        >>> model = Encoder(idim=256)
        >>> model.apply(encoder_init)

    Raises:
        None
    """
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Encoder(torch.nn.Module):
    """
        Singing Tacotron encoder related modules.

    This module contains the implementation of the Encoder class, which is part of
    the Spectrogram prediction network in Singing Tacotron. The encoder converts
    either a sequence of characters or acoustic features into a sequence of hidden
    states.

    The encoder is designed based on the architecture described in
    `Singing-Tacotron: Global Duration Control Attention and Dynamic Filter for
    End-to-end Singing Voice Synthesis`_.

    .. _`Singing-Tacotron: Global Duration Control Attention and Dynamic Filter
    for End-to-end Singing Voice Synthesis`:
       https://arxiv.org/abs/2202.07907

    Attributes:
        idim (int): Dimension of the inputs.
        use_residual (bool): Flag to indicate whether to use residual connections.

    Args:
        idim (int): Dimension of the inputs.
        input_layer (str): Type of input layer, either 'linear' or 'embed'.
        embed_dim (int, optional): Dimension of character embedding. Defaults to 512.
        elayers (int, optional): Number of encoder BLSTM layers. Defaults to 1.
        eunits (int, optional): Number of encoder BLSTM units. Defaults to 512.
        econv_layers (int, optional): Number of encoder convolutional layers.
            Defaults to 3.
        econv_chans (int, optional): Number of encoder convolutional filter
            channels. Defaults to 512.
        econv_filts (int, optional): Size of the encoder convolutional filters.
            Defaults to 5.
        use_batch_norm (bool, optional): Whether to use batch normalization.
            Defaults to True.
        use_residual (bool, optional): Whether to use residual connections.
            Defaults to False.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.5.
        padding_idx (int, optional): Padding index for embeddings. Defaults to 0.

    Returns:
        Tensor: The output of the encoder after forward propagation.

    Raises:
        ValueError: If an unknown input_layer type is provided.

    Examples:
        encoder = Encoder(idim=128, input_layer='embed', embed_dim=256)
        xs, ilens = encoder(torch.randn(32, 100), torch.tensor([100]*32))
    """

    def __init__(
        self,
        idim,
        input_layer="embed",
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize Singing Tacotron encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        if input_layer == "linear":
            self.embed = torch.nn.Linear(idim, econv_chans)
        elif input_layer == "embed":
            self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = (
                    embed_dim if layer == 0 and input_layer == "embed" else econv_chans
                )
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """
                Singing Tacotron encoder related modules.

        This module includes the encoder for the Spectrogram prediction network
        in Singing Tacotron, which is described in `Singing-Tacotron: Global
        Duration Control Attention and Dynamic Filter for End-to-end Singing Voice
        Synthesis`_. This encoder converts either a sequence of characters or
        acoustic features into a sequence of hidden states.

        .. _`Singing-Tacotron: Global Duration Control Attention and Dynamic
        Filter for End-to-end Singing Voice Synthesis`:
           https://arxiv.org/abs/2202.07907

        Attributes:
            idim (int): Dimension of the inputs.
            use_residual (bool): Whether to use residual connections.

        Args:
            xs (Tensor): Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states (B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,).

        Examples:
            >>> encoder = Encoder(idim=256)
            >>> input_tensor = torch.randint(0, 256, (10, 20))  # Example input
            >>> lengths = torch.tensor([20] * 10)  # All sequences of length 20
            >>> states, seq_lengths = encoder(input_tensor, lengths)
        """
        xs = xs.transpose(1, 2)
        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs = xs + self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs = pack_padded_sequence(
            xs.transpose(1, 2), ilens.cpu(), batch_first=True, enforce_sorted=False
        )
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x, ilens):
        """
        Perform inference on the input sequence.

        This method processes the input sequence, which can be either character
        IDs or acoustic features, and returns the corresponding encoder states.

        Args:
            x (Tensor): The sequence of character IDs (T,) or acoustic features
                (T, idim * encoder_reduction_factor).
            ilens (LongTensor): Lengths of the input sequences (B,).

        Returns:
            Tensor: The sequences of encoder states (T, eunits).

        Examples:
            >>> encoder = Encoder(idim=256)
            >>> character_ids = torch.tensor([1, 2, 3, 0])  # Example IDs
            >>> ilens = torch.tensor([3])  # Length of input
            >>> states = encoder.inference(character_ids, ilens)
            >>> print(states.shape)  # Should output: torch.Size([T, eunits])
        """

        xs = x

        return self.forward(xs, ilens)[0][0]


class Duration_Encoder(torch.nn.Module):
    """
    Duration_Encoder module of Spectrogram prediction network.

    This module is part of the Singing Tacotron architecture. It converts a
    sequence of durations and tempo features into a transition token, which
    is essential for generating singing voice synthesis. The architecture
    follows the principles outlined in the paper `Singing-Tacotron: Global
    Duration Control Attention and Dynamic Filter for End-to-end Singing
    Voice Synthesis`_.

    .. _`Singing-Tacotron: Global Duration Control Attention and Dynamic
    Filter for End-to-end Singing Voice Synthesis`:
       https://arxiv.org/abs/2202.07907

    Attributes:
        idim (int): Dimension of the inputs.

    Args:
        idim (int): Dimension of the inputs.
        embed_dim (int, optional): Dimension of character embedding. Default is 512.
        dropout_rate (float, optional): Dropout rate. Default is 0.5.
        padding_idx (int, optional): Padding index for embedding. Default is 0.

    Examples:
        >>> duration_encoder = Duration_Encoder(idim=10)
        >>> input_tensor = torch.rand(2, 5, 10)  # Batch of 2, Tmax=5, feature_len=10
        >>> output = duration_encoder(input_tensor)
        >>> print(output.shape)  # Should output: torch.Size([2, 5, 1])
    """

    def __init__(
        self,
        idim,
        embed_dim=512,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize Singing-Tacotron encoder module.

        Args:
            idim (int) Dimension of the inputs.
            embed_dim (int, optional) Dimension of character embedding.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Duration_Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim

        # define network layer modules
        self.dense24 = torch.nn.Linear(idim, 24)
        self.convs = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                24,
                32,
                3,
                stride=1,
                bias=False,
                padding=2 // 2,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                32,
                32,
                3,
                stride=1,
                bias=False,
                padding=2 // 2,
            ),
            torch.nn.ReLU(),
        )
        self.dense1 = torch.nn.Linear(32, 1)
        self.nntanh = torch.nn.Tanh()

        # initialize
        self.apply(encoder_init)

    def forward(self, xs):
        """
        Calculate forward propagation.

        This method computes the forward pass of the Duration_Encoder module,
        transforming the input duration sequence into transition tokens.

        Args:
            xs (Tensor): Batch of the duration sequence with shape
                (B, Tmax, feature_len).

        Returns:
            Tensor: Batch of the sequences of transition tokens with shape
                (B, Tmax, 1).
            LongTensor: Batch of lengths of each sequence (B,).

        Examples:
            >>> encoder = Duration_Encoder(idim=10)
            >>> duration_sequence = torch.rand(4, 5, 10)  # Example input
            >>> output = encoder.forward(duration_sequence)
            >>> print(output.shape)
            torch.Size([4, 5, 1])  # Output shape
        """
        xs = self.dense24(xs).transpose(1, 2)
        xs = self.convs(xs).transpose(1, 2)
        xs = self.dense1(xs)
        xs = self.nntanh(xs)
        xs = (xs + 1) / 2
        return xs

    def inference(self, x):
        """
            Inference.

        This method performs inference by processing a sequence of character IDs or
        acoustic features and returning the corresponding encoder states.

        Args:
            x (Tensor): The sequence of character IDs (T,) or acoustic features
                (T, idim * encoder_reduction_factor).
            ilens (LongTensor): The lengths of each input sequence (B,).

        Returns:
            Tensor: The sequences of encoder states (T, eunits).

        Examples:
            >>> encoder = Encoder(idim=40)
            >>> char_ids = torch.tensor([1, 2, 3, 4])
            >>> lengths = torch.tensor([4])
            >>> states = encoder.inference(char_ids, lengths)
            >>> print(states.shape)
            torch.Size([4, 512])  # Assuming eunits is 512
        """

        xs = x

        return self.forward(xs)
