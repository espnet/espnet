# Copyright 2022 Yosuke Higuchi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Masked LM Decoder definition."""
from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class MLMDecoder(AbsDecoder):
    """
    Masked LM Decoder definition for sequence-to-sequence models.

    This class implements a masked language model decoder that utilizes 
    multi-head attention and position-wise feed-forward networks. It is 
    designed to handle the decoding of sequences while incorporating 
    positional encodings and normalization techniques.

    Attributes:
        embed (torch.nn.Sequential): The embedding layer that converts input token 
            IDs to embeddings. Can be an embedding layer or a linear layer.
        normalize_before (bool): Indicates whether to apply normalization before 
            the decoder layers.
        after_norm (LayerNorm, optional): Layer normalization applied after the 
            decoder layers if normalize_before is True.
        output_layer (torch.nn.Linear, optional): Linear layer for output if 
            use_output_layer is True.
        decoders (torch.nn.ModuleList): A list of decoder layers that process the 
            input embeddings and produce output scores.

    Args:
        vocab_size (int): Size of the vocabulary, including a mask token.
        encoder_output_size (int): Size of the encoder output features.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in the feed-forward layers. 
            Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults 
            to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional 
            encodings. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self 
            attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source 
            attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer, either "embed" or 
            "linear". Defaults to "embed".
        use_output_layer (bool, optional): Whether to use an output layer. 
            Defaults to True.
        pos_enc_class (type, optional): Class for positional encoding. Defaults to 
            PositionalEncoding.
        normalize_before (bool, optional): Whether to normalize inputs before 
            passing them to decoder layers. Defaults to True.
        concat_after (bool, optional): Whether to concatenate inputs after 
            attention. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x (torch.Tensor): Decoded token scores before softmax 
            (batch, maxlen_out, vocab_size) if use_output_layer is True.
            - olens (torch.Tensor): Lengths of the output sequences (batch,).

    Raises:
        ValueError: If the input_layer argument is not "embed" or "linear".

    Examples:
        >>> decoder = MLMDecoder(vocab_size=100, encoder_output_size=512)
        >>> hs_pad = torch.randn(32, 10, 512)  # (batch, maxlen_in, feat)
        >>> hlens = torch.tensor([10] * 32)     # (batch)
        >>> ys_in_pad = torch.randint(0, 100, (32, 15))  # (batch, maxlen_out)
        >>> ys_in_lens = torch.tensor([15] * 32)  # (batch)
        >>> output, output_lengths = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        This decoder is typically used in conjunction with an encoder in 
        sequence-to-sequence models for tasks such as automatic speech 
        recognition (ASR) and machine translation.

    Todo:
        - Implement additional features such as attention masking and 
        alternative normalization techniques.
    """
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        vocab_size += 1  # for mask token

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward decoder.

        This method performs the forward pass of the masked language model 
        decoder. It takes the encoded memory from the encoder, input token 
        ids, and their respective lengths to produce decoded token scores 
        before softmax and the output lengths.

        Args:
            hs_pad (torch.Tensor): 
                Encoded memory, shape (batch, maxlen_in, feat) with dtype 
                float32.
            hlens (torch.Tensor): 
                Lengths of the encoded memory, shape (batch).
            ys_in_pad (torch.Tensor): 
                Input token ids, shape (batch, maxlen_out) with dtype int64. 
                If `input_layer` is set to "embed", this should be a tensor of 
                token ids; otherwise, it should be a tensor of shape 
                (batch, maxlen_out, #mels).
            ys_in_lens (torch.Tensor): 
                Lengths of the input sequences, shape (batch).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax, 
                  shape (batch, maxlen_out, token), only if 
                  `use_output_layer` is True.
                - olens (torch.Tensor): Output lengths, shape (batch,).

        Examples:
            >>> decoder = MLMDecoder(vocab_size=100, encoder_output_size=256)
            >>> hs_pad = torch.rand(32, 10, 256)  # (batch, maxlen_in, feat)
            >>> hlens = torch.randint(1, 10, (32,))  # (batch)
            >>> ys_in_pad = torch.randint(0, 100, (32, 15))  # (batch, maxlen_out)
            >>> ys_in_lens = torch.randint(1, 15, (32,))  # (batch)
            >>> output, output_lengths = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)

        Note:
            Ensure that the input tensor shapes are consistent with the 
            specified dimensions, and that the model has been properly 
            initialized before calling this method.

        Raises:
            ValueError: If the input_layer is neither "embed" nor "linear".
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        tgt_max_len = tgt_mask.size(-1)
        # tgt_mask_tmp: (B, L, L)
        tgt_mask_tmp = tgt_mask.transpose(1, 2).repeat(1, 1, tgt_max_len)
        tgt_mask = tgt_mask.repeat(1, tgt_max_len, 1) & tgt_mask_tmp

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens))[:, None, :].to(memory.device)

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens
