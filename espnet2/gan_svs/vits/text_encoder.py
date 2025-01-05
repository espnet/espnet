# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text encoder module in VISinger.

This code is based on https://github.com/jaywalnut310/vits
and https://github.com/zhangyongmao/VISinger2.

"""

import math
from typing import Optional, Tuple

import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class TextEncoder(torch.nn.Module):
    """
        TextEncoder class for encoding text input in the VISinger model.

    This module implements a text encoder as described in the paper
    `Conditional Variational Autoencoder with Adversarial Learning for
    End-to-End Text-to-Speech`_. It utilizes a conformer architecture, which
    incorporates convolutional layers in addition to the standard attention
    mechanism.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    Attributes:
        attention_dim (int): The dimension of attention.
        encoder (Encoder): The conformer-based encoder module.
        emb_phone_dim (int): The dimension of phone embeddings.
        emb_phone (torch.nn.Embedding): Embedding layer for phone inputs.
        emb_pitch_dim (int): The dimension of pitch embeddings.
        emb_pitch (torch.nn.Embedding): Embedding layer for pitch inputs.
        emb_slur (Optional[torch.nn.Embedding]): Embedding layer for slur inputs.
        emb_dur (torch.nn.Linear): Linear layer for duration inputs.
        pre_net (torch.nn.Linear): Preprocessing layer for the main input.
        pre_dur_net (torch.nn.Linear): Preprocessing layer for duration input.
        proj (torch.nn.Conv1d): Convolutional layer for projection.
        proj_pitch (torch.nn.Conv1d): Convolutional layer for pitch projection.

    Args:
        vocabs (int): Vocabulary size.
        attention_dim (int): Dimension of the attention mechanism.
        attention_heads (int): Number of attention heads.
        linear_units (int): Number of linear units in positionwise layers.
        blocks (int): Number of encoder blocks.
        positionwise_layer_type (str): Type of positionwise layer.
        positionwise_conv_kernel_size (int): Kernel size for positionwise layers.
        positional_encoding_layer_type (str): Type of positional encoding layer.
        self_attention_layer_type (str): Type of self-attention layer.
        activation_type (str): Type of activation function.
        normalize_before (bool): Whether to apply LayerNorm before attention.
        use_macaron_style (bool): Whether to use Macaron-style components.
        use_conformer_conv (bool): Whether to use convolution in conformer.
        conformer_kernel_size (int): Kernel size for conformer convolution.
        dropout_rate (float): Dropout rate for layers.
        positional_dropout_rate (float): Dropout rate for positional encoding.
        attention_dropout_rate (float): Dropout rate for attention layers.
        use_slur (bool): Whether to use slur embedding.

    Examples:
        >>> text_encoder = TextEncoder(vocabs=5000)
        >>> phone_tensor = torch.randint(0, 5000, (32, 100))
        >>> phone_lengths = torch.randint(1, 101, (32,))
        >>> midi_tensor = torch.randint(0, 129, (32, 100))
        >>> duration_tensor = torch.rand((32, 100))
        >>> encoded_output = text_encoder(phone_tensor, phone_lengths, midi_tensor,
        ...                                duration_tensor)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Encoded hidden representation (B, attention_dim, T_text).
            - Mask tensor for padded parts (B, 1, T_text).
            - Encoded hidden representation for duration (B, attention_dim, T_text).
            - Encoded hidden representation for pitch (B, attention_dim, T_text).

    Raises:
        ValueError: If the input tensors do not match the expected dimensions.
    """

    def __init__(
        self,
        vocabs: int,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 6,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 3,
        positional_encoding_layer_type: str = "rel_pos",
        self_attention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        normalize_before: bool = True,
        use_macaron_style: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 7,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        use_slur=True,
    ):
        """Initialize TextEncoder module.

        Args:
            vocabs (int): Vocabulary size.
            attention_dim (int): Attention dimension.
            attention_heads (int): Number of attention heads.
            linear_units (int): Number of linear units of positionwise layers.
            blocks (int): Number of encoder blocks.
            positionwise_layer_type (str): Positionwise layer type.
            positionwise_conv_kernel_size (int): Positionwise layer's kernel size.
            positional_encoding_layer_type (str): Positional encoding layer type.
            self_attention_layer_type (str): Self-attention layer type.
            activation_type (str): Activation function type.
            normalize_before (bool): Whether to apply LayerNorm before attention.
            use_macaron_style (bool): Whether to use macaron style components.
            use_conformer_conv (bool): Whether to use conformer conv layers.
            conformer_kernel_size (int): Conformer's conv kernel size.
            dropout_rate (float): Dropout rate.
            positional_dropout_rate (float): Dropout rate for positional encoding.
            attention_dropout_rate (float): Dropout rate for attention.
            use_slur (bool): Whether to use slur embedding.

        """
        super().__init__()
        # store for forward
        self.attention_dim = attention_dim

        # define modules
        self.encoder = Encoder(
            idim=-1,
            input_layer=None,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            normalize_before=normalize_before,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=positional_encoding_layer_type,
            selfattention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_conformer_conv,
            cnn_module_kernel=conformer_kernel_size,
        )
        self.emb_phone_dim = 256
        self.emb_phone = torch.nn.Embedding(vocabs, self.emb_phone_dim)
        torch.nn.init.normal_(self.emb_phone.weight, 0.0, self.emb_phone_dim**-0.5)

        self.emb_pitch_dim = 128
        self.emb_pitch = torch.nn.Embedding(
            129, self.emb_pitch_dim
        )  # Should we count the number of midis instead of 129?
        torch.nn.init.normal_(self.emb_pitch.weight, 0.0, self.emb_pitch_dim**-0.5)

        if use_slur:
            self.emb_slur = torch.nn.Embedding(2, 64)
            torch.nn.init.normal_(self.emb_slur.weight, 0.0, 64**-0.5)

        if use_slur:
            self.emb_dur = torch.nn.Linear(1, 64)
        else:
            self.emb_dur = torch.nn.Linear(1, 128)

        self.pre_net = torch.nn.Linear(512, attention_dim)
        self.pre_dur_net = torch.nn.Linear(512, attention_dim)

        self.proj = torch.nn.Conv1d(attention_dim, attention_dim, 1)
        self.proj_pitch = torch.nn.Conv1d(self.emb_pitch_dim, attention_dim, 1)

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        midi_id: torch.Tensor,
        dur: torch.Tensor,
        slur: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Text encoder module in VISinger.

        This is a module of text encoder described in `Conditional Variational Autoencoder
        with Adversarial Learning for End-to-End Text-to-Speech`_.

        Instead of the relative positional Transformer, we use conformer architecture as
        the encoder module, which contains additional convolution layers.

        .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
            Text-to-Speech`: https://arxiv.org/abs/2006.04558

        Args:
                phone (Tensor): Input index tensor (B, T_text).
                phone_lengths (Tensor): Length tensor (B,).
                midi_id (Tensor): Input midi tensor (B, T_text).
                dur (Tensor): Input duration tensor (B, T_text).

        Examples:
            >>> encoder = TextEncoder(vocabs=1000)
            >>> phone_tensor = torch.randint(0, 1000, (2, 10))
            >>> phone_lengths = torch.tensor([10, 8])
            >>> midi_id = torch.randint(0, 129, (2, 10))
            >>> dur = torch.rand((2, 10))
            >>> output = encoder(phone_tensor, phone_lengths, midi_id, dur)
            >>> print(output)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Encoded hidden representation (B, attention_dim, T_text),
                Mask tensor for padded part (B, 1, T_text),
                Encoded hidden representation for duration (B, attention_dim, T_text),
                Encoded hidden representation for pitch (B, attention_dim, T_text).
        """
        phone_end = self.emb_phone(phone) * math.sqrt(self.emb_phone_dim)
        pitch_end = self.emb_pitch(midi_id) * math.sqrt(self.emb_pitch_dim)

        if slur is not None:
            slur_end = self.emb_slur(slur) * math.sqrt(64)

        dur = dur.float()
        dur_end = self.emb_dur(dur.unsqueeze(-1))

        if slur is not None:
            x = torch.cat([phone_end, pitch_end, slur_end, dur_end], dim=-1)
        else:
            x = torch.cat([phone_end, pitch_end, dur_end], dim=-1)

        dur_input = self.pre_dur_net(x)
        dur_input = torch.transpose(dur_input, 1, -1)

        x = self.pre_net(x)
        # x = torch.transpose(x, 1, -1)  # [b, h, t]

        x_mask = (
            make_non_pad_mask(phone_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )
        # encoder assume the channel last (B, T_text, attention_dim)
        # but mask shape shoud be (B, 1, T_text)
        x, _ = self.encoder(x, x_mask)

        # convert the channel first to (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        x = self.proj(x) * x_mask

        pitch_info = self.proj_pitch(pitch_end.transpose(1, 2))

        return x, x_mask, dur_input, pitch_info
