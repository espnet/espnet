# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Translatotron2 related modules for ESPnet2."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
    DurationPredictor as FastDurationPredictor,
)


class Translatotron2(AbsSynthesizer):
    """
        Translatotron2 module.

    This is a module of the synthesizer in Translatotron2 described in
    `Translatotron 2: High-quality direct speech-to-speech translation with
    voice preservation`_.

    Attributes:
        idim (int): Input dimension for the model.
        odim (int): Output dimension for the model.
        synthesizer_type (str): Type of synthesizer (default: "rnn").
        layers (int): Number of layers in the synthesizer (default: 2).
        units (int): Number of units in each layer (default: 1024).
        prenet_layers (int): Number of layers in the prenet (default: 2).
        prenet_units (int): Number of units in the prenet (default: 128).
        prenet_dropout_rate (float): Dropout rate for the prenet (default: 0.5).
        postnet_layers (int): Number of layers in the postnet (default: 5).
        postnet_chans (int): Number of channels in the postnet (default: 512).
        postnet_dropout_rate (float): Dropout rate for the postnet (default: 0.5).
        adim (int): Dimension of the attention mechanism (default: 384).
        aheads (int): Number of attention heads (default: 4).
        conformer_rel_pos_type (str): Type of relative positional encoding (default: "legacy").
        conformer_pos_enc_layer_type (str): Layer type for positional encoding (default: "rel_pos").
        conformer_self_attn_layer_type (str): Layer type for self-attention (default: "rel_selfattn").
        conformer_activation_type (str): Activation function type (default: "swish").
        use_macaron_style_in_conformer (bool): Whether to use Macaron style in conformer (default: True).
        use_cnn_in_conformer (bool): Whether to use CNN in conformer (default: True).
        zero_triu (bool): Whether to zero out the upper triangular part of the attention matrix (default: False).
        conformer_enc_kernel_size (int): Kernel size for the conformer encoder (default: 7).
        conformer_dec_kernel_size (int): Kernel size for the conformer decoder (default: 31).
        duration_predictor_layers (int): Number of layers in the duration predictor (default: 2).
        duration_predictor_type (str): Type of duration predictor (default: "rnn").
        duration_predictor_units (int): Number of units in the duration predictor (default: 128).
        spks (Optional[int]): Number of speakers (default: None).
        langs (Optional[int]): Number of languages (default: None).
        spk_embed_dim (Optional[int]): Dimension of speaker embedding (default: None).
        spk_embed_integration_type (str): Type of speaker embedding integration (default: "add").
        init_type (str): Initialization type for the model (default: "xavier_uniform").
        init_enc_alpha (float): Initialization alpha for encoder (default: 1.0).
        init_dec_alpha (float): Initialization alpha for decoder (default: 1.0).
        use_masking (bool): Whether to use masking during training (default: False).
        use_weighted_masking (bool): Whether to use weighted masking during training (default: False).

    Args:
        idim (int): Input dimension for the model.
        odim (int): Output dimension for the model.
        synthesizer_type (str, optional): Type of synthesizer (default: "rnn").
        layers (int, optional): Number of layers in the synthesizer (default: 2).
        units (int, optional): Number of units in each layer (default: 1024).
        prenet_layers (int, optional): Number of layers in the prenet (default: 2).
        prenet_units (int, optional): Number of units in the prenet (default: 128).
        prenet_dropout_rate (float, optional): Dropout rate for the prenet (default: 0.5).
        postnet_layers (int, optional): Number of layers in the postnet (default: 5).
        postnet_chans (int, optional): Number of channels in the postnet (default: 512).
        postnet_dropout_rate (float, optional): Dropout rate for the postnet (default: 0.5).
        adim (int, optional): Dimension of the attention mechanism (default: 384).
        aheads (int, optional): Number of attention heads (default: 4).
        conformer_rel_pos_type (str, optional): Type of relative positional encoding (default: "legacy").
        conformer_pos_enc_layer_type (str, optional): Layer type for positional encoding (default: "rel_pos").
        conformer_self_attn_layer_type (str, optional): Layer type for self-attention (default: "rel_selfattn").
        conformer_activation_type (str, optional): Activation function type (default: "swish").
        use_macaron_style_in_conformer (bool, optional): Whether to use Macaron style in conformer (default: True).
        use_cnn_in_conformer (bool, optional): Whether to use CNN in conformer (default: True).
        zero_triu (bool, optional): Whether to zero out the upper triangular part of the attention matrix (default: False).
        conformer_enc_kernel_size (int, optional): Kernel size for the conformer encoder (default: 7).
        conformer_dec_kernel_size (int, optional): Kernel size for the conformer decoder (default: 31).
        duration_predictor_layers (int, optional): Number of layers in the duration predictor (default: 2).
        duration_predictor_type (str, optional): Type of duration predictor (default: "rnn").
        duration_predictor_units (int, optional): Number of units in the duration predictor (default: 128).
        spks (Optional[int], optional): Number of speakers (default: None).
        langs (Optional[int], optional): Number of languages (default: None).
        spk_embed_dim (Optional[int], optional): Dimension of speaker embedding (default: None).
        spk_embed_integration_type (str, optional): Type of speaker embedding integration (default: "add").
        init_type (str, optional): Initialization type for the model (default: "xavier_uniform").
        init_enc_alpha (float, optional): Initialization alpha for encoder (default: 1.0).
        init_dec_alpha (float, optional): Initialization alpha for decoder (default: 1.0).
        use_masking (bool, optional): Whether to use masking during training (default: False).
        use_weighted_masking (bool, optional): Whether to use weighted masking during training (default: False).

    Returns:
        None

    Examples:
        >>> model = Translatotron2(idim=80, odim=80)
        >>> print(model)
        Translatotron2(...)

    Note:
        This class is part of the ESPnet2 speech synthesis framework.
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        synthesizer_type: str = "rnn",
        layers: int = 2,
        units: int = 1024,
        # for prenet
        prenet_layers: int = 2,
        prenet_units: int = 128,
        prenet_dropout_rate: float = 0.5,
        # for postnet
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_dropout_rate: float = 0.5,
        # for transformer
        adim: int = 384,
        aheads: int = 4,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # duration predictor
        duration_predictor_layers: int = 2,
        duration_predictor_type: str = "rnn",
        duration_predictor_units: int = 128,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        return


class Prenet(nn.Module):
    """
        Non-Attentive Tacotron (NAT) Prenet.

    The Prenet is a feedforward neural network module that acts as a
    preprocessing layer for the inputs in the NAT framework. It consists of
    multiple linear layers followed by dropout and ReLU activation. This
    module helps in learning the representation of the input features
    before passing them to the main synthesizer.

    Attributes:
        layers (ModuleList): A list of linear layers for processing input.
        dropout (Dropout): Dropout layer to prevent overfitting.
        activation (ReLU): Activation function applied after each layer.

    Args:
        idim (int): The dimension of the input features.
        units (int, optional): The number of units in each layer. Default is 128.
        num_layers (int, optional): The number of layers in the Prenet. Default is 2.
        dropout (float, optional): The dropout rate applied after each layer.
            Default is 0.5.

    Examples:
        >>> prenet = Prenet(idim=256, units=128, num_layers=2, dropout=0.5)
        >>> input_tensor = torch.randn(10, 256)  # Batch size of 10
        >>> output = prenet(input_tensor)
        >>> output.shape
        torch.Size([10, 128])  # Output shape after processing
    """

    def __init__(self, idim, units=128, num_layers=2, dropout=0.5):
        super(Prenet, self).__init__()
        sizes = [units] * num_layers
        in_sizes = [idim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
            Pass the input through the Prenet layers.

        The forward method applies a series of linear transformations followed by
        dropout and ReLU activation to the input tensor `x`. Each layer is defined
        in the constructor and is executed sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor after passing through the Prenet layers
            with shape [batch_size, units].

        Examples:
            >>> prenet = Prenet(idim=256, units=128, num_layers=2, dropout=0.5)
            >>> input_tensor = torch.randn(10, 256)  # Example batch of size 10
            >>> output_tensor = prenet(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 128])

        Note:
            The input tensor should be of the appropriate shape and dimension
            matching the expected input for the Prenet layers.
        """
        for linear in self.layers:
            x = self.dropout(self.activation(linear(x)))
        return x


class DurationPredictor(nn.Module):
    """
        Non-Attentive Tacotron (NAT) Duration Predictor module.

    This module predicts the duration of phonemes based on the encoder outputs.
    It utilizes a bidirectional LSTM to model the temporal dependencies in the
    input sequences, followed by a linear layer to output the duration predictions.

    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM layer for duration prediction.
        proj (nn.LinearNorm): Linear layer to project LSTM outputs to duration values.
        relu (nn.ReLU): ReLU activation function.

    Args:
        cfg: Configuration object containing parameters for the duration predictor.
            Expected attributes in cfg include:
                - units (int): Number of input features for the LSTM.
                - duration_lstm_dim (int): Dimension of the LSTM output features.

    Returns:
        Tensor: Duration predictions of shape [batch_size, hidden_length].

    Raises:
        ValueError: If the input lengths are not compatible with the encoder outputs.

    Examples:
        >>> predictor = DurationPredictor(cfg)
        >>> encoder_outputs = torch.randn(16, 50, cfg.units)  # Example tensor
        >>> durations = predictor(encoder_outputs)
        >>> print(durations.shape)  # Output shape will be [16, 50]
    """

    def __init__(self, cfg):
        super(FastDurationPredictor, self).__init__()

        self.lstm = nn.LSTM(
            cfg.units,
            int(cfg.duration_lstm_dim / 2),
            2,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.LinearNorm(cfg.duration_lstm_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_outputs, input_lengths=None):
        """
        Forward Duration Predictor.

        This method processes the encoder outputs through an LSTM layer to predict
        the duration of each phoneme. It optionally accepts input lengths to
        remove padding from the encoder outputs.

        Args:
            encoder_outputs (torch.Tensor): A tensor of shape
                [batch_size, hidden_length, encoder_lstm_dim] representing
                the outputs from the encoder.
            input_lengths (torch.Tensor, optional): A tensor of shape
                [batch_size] that specifies the actual lengths of the inputs
                for each batch, used to handle padding. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, hidden_length] that
            contains the predicted durations for each phoneme.

        Examples:
            >>> duration_predictor = DurationPredictor(cfg)
            >>> encoder_outputs = torch.randn(16, 50, 256)  # Example tensor
            >>> input_lengths = torch.tensor([50] * 16)  # No padding
            >>> durations = duration_predictor(encoder_outputs, input_lengths)
            >>> print(durations.shape)  # Output: torch.Size([16, 50])
        """

        batch_size = encoder_outputs.size(0)
        hidden_length = encoder_outputs.size(1)

        # remove pad activations
        if input_lengths is not None:
            encoder_outputs = pack_padded_sequence(
                encoder_outputs, input_lengths, batch_first=True, enforce_sorted=False
            )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(encoder_outputs)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        outputs = self.relu(self.proj(outputs))

        return outputs.view(batch_size, hidden_length)


class GaussianUpsampling(nn.Module):
    """
        Gaussian Upsample.

    This module implements Gaussian upsampling for the Non-Attentive Tacotron.
    It is part of the synthesizer in the ExpressiveTacotron project.

    References:
    - Non-attention Tacotron: https://arxiv.org/abs/2010.04301
    - ExpressiveTacotron: https://github.com/BridgetteSong/ExpressiveTacotron/

    Attributes:
        mask_score (float): A constant used to mask out irrelevant weights during
            the softmax operation.

    Methods:
        forward(encoder_outputs, durations, vars, input_lengths=None):
            Performs Gaussian upsampling on the provided encoder outputs.

    Args:
        encoder_outputs (torch.Tensor): The encoder outputs of shape
            [batch_size, hidden_length, dim].
        durations (torch.Tensor): The phoneme durations of shape
            [batch_size, hidden_length].
        vars (torch.Tensor): The phoneme attended ranges of shape
            [batch_size, hidden_length].
        input_lengths (torch.Tensor, optional): The lengths of the inputs of
            shape [batch_size]. Defaults to None.

    Returns:
        torch.Tensor: The upsampled encoder outputs of shape
            [batch_size, frame_length, dim].

    Examples:
        >>> gaussian_upsample = GaussianUpsampling()
        >>> encoder_outputs = torch.randn(2, 5, 256)  # Example tensor
        >>> durations = torch.tensor([[1, 2, 1, 1, 1], [2, 1, 2, 1, 1]])
        >>> vars = torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.1], [0.2, 0.1, 0.2, 0.1, 0.1]])
        >>> output = gaussian_upsample(encoder_outputs, durations, vars)
        >>> output.shape
        torch.Size([2, frame_length, 256])
    """

    def __init__(self):
        super(GaussianUpsampling, self).__init__()
        self.mask_score = -1e15

    def forward(self, encoder_outputs, durations, vars, input_lengths=None):
        """
                Gaussian Upsample.

        Non-attention Tacotron:
            - https://arxiv.org/abs/2010.04301
        This source code is an implementation of the ExpressiveTacotron from
        BridgetteSong:
            - https://github.com/BridgetteSong/ExpressiveTacotron/

        Attributes:
            mask_score (float): A constant used for masking scores.

        Args:
            encoder_outputs (torch.Tensor): Encoder outputs with shape
                [batch_size, hidden_length, dim].
            durations (torch.Tensor): Phoneme durations with shape
                [batch_size, hidden_length].
            vars (torch.Tensor): Phoneme attended ranges with shape
                [batch_size, hidden_length].
            input_lengths (torch.Tensor, optional): Lengths of input sequences with
                shape [batch_size]. Defaults to None.

        Returns:
            torch.Tensor: Upsampled encoder outputs with shape
                [batch_size, frame_length, dim].

        Examples:
            >>> model = GaussianUpsampling()
            >>> encoder_outputs = torch.rand(2, 5, 128)  # Example tensor
            >>> durations = torch.tensor([[1, 2, 1, 3, 1], [1, 1, 1, 1, 1]])
            >>> vars = torch.rand(2, 5)
            >>> output = model(encoder_outputs, durations, vars)
            >>> print(output.shape)  # Output shape will be [2, frame_length, 128]

        Note:
            The input_lengths argument is optional and can be used to apply
            masking to the upsampling process.
        """
        batch_size = encoder_outputs.size(0)
        hidden_length = encoder_outputs.size(1)
        frame_length = int(torch.sum(durations, dim=1).max().item())
        c = torch.cumsum(durations, dim=1).float() - 0.5 * durations
        c = c.unsqueeze(2)  # [batch_size, hidden_length, 1]
        t = (
            torch.arange(frame_length, device=encoder_outputs.device)
            .expand(batch_size, hidden_length, frame_length)
            .float()
        )  # [batch_size, hidden_length, frame_length]
        vars = vars.view(batch_size, -1, 1)  # [batch_size, hidden_length, 1]

        w_t = -0.5 * (
            np.log(2.0 * np.pi) + torch.log(vars) + torch.pow(t - c, 2) / vars
        )  # [batch_size, hidden_length, frame_length]

        if input_lengths is not None:
            input_masks = ~self.get_mask_from_lengths(
                input_lengths, hidden_length
            )  # [batch_size, hidden_length]
            input_masks = torch.tensor(input_masks, dtype=torch.bool, device=w_t.device)
            masks = input_masks.unsqueeze(2)
            w_t.data.masked_fill_(masks, self.mask_score)
        w_t = F.softmax(w_t, dim=1)

        encoder_upsampling_outputs = torch.bmm(
            w_t.transpose(1, 2), encoder_outputs
        )  # [batch_size, frame_length, encoder_hidden_size]

        return encoder_upsampling_outputs

    def get_mask_from_lengths(self, lengths, max_len=None):
        """
        Generate a mask from lengths.

        This method creates a boolean mask array that indicates which
        positions in the sequence should be considered valid based on
        the provided lengths. The mask has a shape of (batch_size,
        max_len), where `max_len` is the maximum length specified or
        the maximum length found in the `lengths` array.

        Args:
            lengths (np.ndarray): An array of shape (batch_size,)
                containing the valid lengths for each sequence in the
                batch.
            max_len (Optional[int]): The maximum length for the mask.
                If not provided, it will be set to the maximum value
                in `lengths`.

        Returns:
            np.ndarray: A boolean array of shape (batch_size, max_len)
                where each row corresponds to a sequence and contains
                `True` for valid positions and `False` for padding
                positions.

        Examples:
            >>> lengths = np.array([3, 5, 2])
            >>> mask = get_mask_from_lengths(lengths)
            >>> print(mask)
            [[ True  True  True]
             [ True  True  True  True  True]
             [ True  True]]

            >>> mask_with_max_len = get_mask_from_lengths(lengths, max_len=5)
            >>> print(mask_with_max_len)
            [[ True  True  True False False]
             [ True  True  True  True  True]
             [ True  True False False False]]

        Note:
            The returned mask can be used to filter out padding
            values in sequence data during model training or
            inference.
        """
        if max_len is None:
            max_len = max(lengths)
        ids = np.arange(0, max_len)
        mask = ids < lengths.reshape(-1, 1)
        return mask
