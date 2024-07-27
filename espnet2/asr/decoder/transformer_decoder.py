# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any, List, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import (
    BatchScorerInterface,
    MaskParallelScorerInterface,
)


class BaseTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """
        Base class for Transformer decoder modules.

    This abstract base class provides the foundation for implementing various
    Transformer decoder architectures. It defines the common structure and
    methods that all Transformer decoder variants should implement.

    Attributes:
        embed (torch.nn.Sequential): The embedding layer for input tokens.
        decoders (torch.nn.ModuleList): List of decoder layers (to be implemented by subclasses).
        after_norm (LayerNorm): Layer normalization applied after the decoder stack.
        output_layer (torch.nn.Linear): Linear layer for final output projection.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate for positional encoding.
        input_layer (str): Type of input layer ('embed' or 'linear').
        use_output_layer (bool): Whether to use an output layer.
        pos_enc_class: Positional encoding class to use.
        normalize_before (bool): Whether to apply layer normalization before each block.

    Note:
        Subclasses should implement the specific decoder architecture by
        defining the `decoders` attribute.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        super().__init__()
        attention_dim = encoder_output_size

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

        self._output_size_bf_softmax = attention_dim
        # Must set by the inheritance
        self.decoders = None
        self.batch_ids = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass of the decoder.

        This method processes the encoder output and generates decoded sequences.

        Args:
            hs_pad (torch.Tensor): Encoded memory, shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of encoded sequences, shape (batch,).
            ys_in_pad (torch.Tensor): Input token ids or features, shape (batch, maxlen_out).
                If input_layer is "embed", it contains token ids.
                Otherwise, it contains input features.
            ys_in_lens (torch.Tensor): Lengths of input sequences, shape (batch,).
            return_hs (bool, optional): Whether to return the last hidden state. Defaults to False.
            return_all_hs (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax,
                    shape (batch, maxlen_out, vocab_size) if use_output_layer is True.
                - olens (torch.Tensor): Output lengths, shape (batch,).
                - hidden (torch.Tensor, optional): Last hidden state if return_hs is True.
                - intermediate_outs (List[torch.Tensor], optional): All intermediate hidden states
                    if return_all_hs is True.

        Note:
            The behavior of this method can be customized using the return_hs and
            return_all_hs flags to obtain additional hidden state information.
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        x = self.embed(tgt)
        intermediate_outs = []
        for layer_idx, decoder_layer in enumerate(self.decoders):
            x, tgt_mask, memory, memory_mask = decoder_layer(
                x, tgt_mask, memory, memory_mask
            )
            if return_all_hs:
                intermediate_outs.append(x)
        if self.normalize_before:
            x = self.after_norm(x)
        if return_hs:
            hidden = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        if return_hs:
            return (x, hidden), olens
        elif return_all_hs:
            return (x, intermediate_outs), olens
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
                Perform one step of the decoder forward pass.

        This method is typically used for incremental decoding, processing one token at a time.

        Args:
            tgt (torch.Tensor): Input token ids, shape (batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask, shape (batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2-, dtype=torch.bool in PyTorch 1.2+ (including 1.2).
            memory (torch.Tensor): Encoded memory, shape (batch, maxlen_in, feat).
            memory_mask (torch.Tensor, optional): Encoded memory mask, shape (batch, 1, maxlen_in).
            cache (List[torch.Tensor], optional): Cached output list of shape (batch, max_time_out-1, size).
            return_hs (bool, optional): Whether to return the hidden state. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - y (torch.Tensor): Output tensor of shape (batch, maxlen_out, token)
                    if use_output_layer is True, else the last hidden state.
                - new_cache (List[torch.Tensor]): Updated cache for each decoder layer.
                - hidden (torch.Tensor, optional): Hidden state if return_hs is True.

        Note:
            This method is crucial for efficient autoregressive decoding in inference time.
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if return_hs:
            hidden = y
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return (y, hidden), new_cache
        return y, new_cache

    def score(self, ys, state, x, return_hs=False):
        """
                Score a sequence of tokens.

        This method computes the log probability score for a given sequence of tokens.

        Args:
            ys (torch.Tensor): Sequence of tokens to score, shape (sequence_length,).
            state (List[Any]): Previous decoder state.
            x (torch.Tensor): Encoder output, shape (1, encoder_output_size).
            return_hs (bool, optional): Whether to return the hidden state. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - logp (torch.Tensor): Log probability scores for the input sequence,
                    shape (sequence_length, vocab_size).
                - state (List[Any]): Updated decoder state.
                - hs (torch.Tensor, optional): Hidden state if return_hs is True.

        Note:
            This method is typically used in beam search and other decoding algorithms
            to evaluate and rank candidate sequences.
        """
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if return_hs:
            (logp, hs), state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), hs, state
        else:
            logp, state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
                Score a batch of token sequences.

        This method computes the log probability scores for a batch of token sequences.

        Args:
            ys (torch.Tensor): Batch of token sequences, shape (batch_size, sequence_length).
            states (List[Any]): List of scorer states for prefix tokens.
            xs (torch.Tensor): Batch of encoder outputs, shape (batch_size, max_length, encoder_output_size).
            return_hs (bool, optional): Whether to return the hidden states. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - logp (torch.Tensor): Batch of log probability scores for the next token,
                    shape (batch_size, vocab_size).
                - state_list (List[List[Any]]): Updated list of scorer states for each sequence in the batch.
                - hs (torch.Tensor, optional): Hidden states if return_hs is True.

        Note:
            This method is optimized for batch processing, which is more efficient than
            scoring sequences individually, especially during beam search or batch decoding.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        if return_hs:
            (logp, hs), states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=return_hs
            )
        else:
            logp, states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=return_hs
            )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        if return_hs:
            return (logp, hs), state_list
        return logp, state_list

    def forward_partially_AR(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_lengths: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (n_mask * n_beam, maxlen_out)
            tgt_mask: input token mask,  (n_mask * n_beam, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            tgt_lengths: (n_mask * n_beam, )
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)  # (n_mask * n_beam, maxlen_out, D)
        new_cache = []
        if cache is None:
            cache = [None] * len(self.decoders)

        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, tgt_lengths, memory, memory_mask = (
                decoder.forward_partially_AR(
                    x, tgt_mask, tgt_lengths, memory, None, cache=c
                )
            )
            new_cache.append(x)

        if self.batch_ids is None or len(self.batch_ids) < x.size(0):
            self.batch_ids = torch.arange(x.size(0), device=x.device)

        if self.normalize_before:
            y = self.after_norm(
                x[self.batch_ids[: x.size(0)], tgt_lengths.unsqueeze(0) - 1].squeeze(0)
            )
        else:
            y = x[self.batch_ids, tgt_lengths.unsqueeze(0) - 1].squeeze(0)

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, torch.stack(new_cache)

    def batch_score_partially_AR(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        yseq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        # merge states
        if states[0] is None:
            batch_state = None
        else:
            # reshape state of [mask * batch, layer, 1, D]
            # into [layer, mask * batch, 1, D]
            batch_state = states.transpose(0, 1)

        # batch decoding
        tgt_mask = (~make_pad_mask(yseq_lengths)[:, None, :]).to(xs.device)
        m = subsequent_mask(tgt_mask.size(-1), device=xs.device).unsqueeze(0)
        tgt_mask = tgt_mask & m

        logp, states = self.forward_partially_AR(
            ys, tgt_mask, yseq_lengths, xs, cache=batch_state
        )

        # states is torch.Tensor, where shape is (layer, n_mask * n_beam, yseq_len, D)
        # reshape state to [n_mask * n_beam, layer, yseq_len, D]
        state_list = states.transpose(0, 1)
        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    """
        Transformer decoder module.

    This class implements the standard Transformer decoder architecture as described
    in "Attention Is All You Need" (Vaswani et al., 2017). It consists of multiple
    stacked self-attention and encoder-decoder attention layers, followed by a
    feed-forward network.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Defaults to "embed".
        use_output_layer (bool, optional): Whether to use output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        layer_drop_rate (float, optional): Layer dropout rate. Defaults to 0.0.

    Attributes:
        decoders (torch.nn.ModuleList): List of decoder layers.

    Note:
        This implementation allows for easy modification of various hyperparameters
        and architectural choices, making it suitable for a wide range of sequence
        generation tasks.
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
        layer_drop_rate: float = 0.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
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
            layer_drop_rate,
        )


class LightweightConvolutionTransformerDecoder(BaseTransformerDecoder):
    """
        Lightweight Convolution Transformer decoder module.

    This class implements a Transformer decoder that replaces the self-attention
    mechanism with lightweight convolution, as described in "Pay Less Attention
    with Lightweight and Dynamic Convolutions" (Wu et al., 2019). It combines
    the benefits of convolutional neural networks and self-attention.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Defaults to "embed".
        use_output_layer (bool, optional): Whether to use output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        conv_wshare (int, optional): Weight sharing factor for convolution. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): Kernel size for each convolution layer. Defaults to (11, 11, 11, 11, 11, 11).
        conv_usebias (bool, optional): Whether to use bias in convolution layers. Defaults to False.

    Attributes:
        decoders (torch.nn.ModuleList): List of decoder layers with lightweight convolution.

    Note:
        This decoder variant can be more efficient than standard self-attention
        for certain tasks, especially those involving long sequences.
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
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
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


class LightweightConvolution2DTransformerDecoder(BaseTransformerDecoder):
    """
        Lightweight 2D Convolution Transformer decoder module.

    This class implements a Transformer decoder that uses 2D lightweight convolutions
    instead of self-attention. It extends the concept introduced in "Pay Less Attention
    with Lightweight and Dynamic Convolutions" (Wu et al., 2019) to 2D convolutions,
    potentially capturing more complex patterns in the input.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Defaults to "embed".
        use_output_layer (bool, optional): Whether to use output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        conv_wshare (int, optional): Weight sharing factor for 2D convolution. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): Kernel size for each 2D convolution layer. Defaults to (11, 11, 11, 11, 11, 11).
        conv_usebias (bool, optional): Whether to use bias in 2D convolution layers. Defaults to False.

    Attributes:
        decoders (torch.nn.ModuleList): List of decoder layers with 2D lightweight convolution.

    Note:
        This decoder variant may be particularly effective for tasks where the input has
        a 2D structure or where capturing 2D patterns is beneficial.
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
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
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


class DynamicConvolutionTransformerDecoder(BaseTransformerDecoder):
    """
        Dynamic Convolution Transformer decoder module.

    This class implements a Transformer decoder that replaces the self-attention
    mechanism with dynamic convolution, as introduced in "Pay Less Attention with
    Lightweight and Dynamic Convolutions" (Wu et al., 2019). Dynamic convolution
    adapts its weights based on the input, allowing for more flexible and
    context-dependent processing.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Defaults to "embed".
        use_output_layer (bool, optional): Whether to use output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        conv_wshare (int, optional): Weight sharing factor for dynamic convolution. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): Kernel size for each dynamic convolution layer. Defaults to (11, 11, 11, 11, 11, 11).
        conv_usebias (bool, optional): Whether to use bias in dynamic convolution layers. Defaults to False.

    Attributes:
        decoders (torch.nn.ModuleList): List of decoder layers with dynamic convolution.

    Note:
        Dynamic convolution can be more effective than standard self-attention or
        lightweight convolution for certain tasks, especially those requiring
        adaptive processing of input sequences.
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
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
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


class DynamicConvolution2DTransformerDecoder(BaseTransformerDecoder):
    """
        Dynamic 2D Convolution Transformer decoder module.

    This class implements a Transformer decoder that uses 2D dynamic convolutions
    instead of self-attention. It extends the concept of dynamic convolutions
    introduced in "Pay Less Attention with Lightweight and Dynamic Convolutions"
    (Wu et al., 2019) to 2D, allowing for more complex and adaptive processing
    of input sequences with potential 2D structure.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Defaults to "embed".
        use_output_layer (bool, optional): Whether to use output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        conv_wshare (int, optional): Weight sharing factor for 2D dynamic convolution. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): Kernel size for each 2D dynamic convolution layer. Defaults to (11, 11, 11, 11, 11, 11).
        conv_usebias (bool, optional): Whether to use bias in 2D dynamic convolution layers. Defaults to False.

    Attributes:
        decoders (torch.nn.ModuleList): List of decoder layers with 2D dynamic convolution.

    Note:
        This decoder variant may be particularly effective for tasks where the input
        has a 2D structure or where capturing adaptive 2D patterns is beneficial.
        It combines the flexibility of dynamic convolutions with 2D processing capabilities.
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
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
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


class TransformerMDDecoder(BaseTransformerDecoder):
    """
        Transformer Multi-Decoder (MD) module.

    This class implements a Transformer decoder with an additional attention mechanism
    for speech input, making it suitable for multi-modal tasks such as speech translation
    or speech recognition with auxiliary text input.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Defaults to "embed".
        use_output_layer (bool, optional): Whether to use output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.
        use_speech_attn (bool, optional): Whether to use additional attention for speech input. Defaults to True.

    Attributes:
        decoders (torch.nn.ModuleList): List of decoder layers with additional speech attention mechanism.
        use_speech_attn (bool): Indicates whether speech attention is used.

    Note:
        This decoder is designed for tasks that involve both text and speech inputs,
        allowing for more effective integration of multi-modal information. The additional
        speech attention mechanism can be toggled on or off based on the task requirements.
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
        use_speech_attn: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
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
                (
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    )
                    if use_speech_attn
                    else None
                ),
            ),
        )

        self.use_speech_attn = use_speech_attn

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        speech: torch.Tensor = None,
        speech_lens: torch.Tensor = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass of the TransformerMDDecoder.

        This method processes the encoder output, speech input (if applicable), and generates
        decoded sequences, optionally returning intermediate hidden states.

        Args:
            hs_pad (torch.Tensor): Encoded memory, shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of encoded sequences, shape (batch,).
            ys_in_pad (torch.Tensor): Input token ids or features, shape (batch, maxlen_out).
                If input_layer is "embed", it contains token ids.
                Otherwise, it contains input features.
            ys_in_lens (torch.Tensor): Lengths of input sequences, shape (batch,).
            speech (torch.Tensor, optional): Speech input tensor, shape (batch, speech_maxlen, speech_feat).
            speech_lens (torch.Tensor, optional): Lengths of speech input sequences, shape (batch,).
            return_hs (bool, optional): Whether to return the last hidden state. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax,
                    shape (batch, maxlen_out, vocab_size) if use_output_layer is True.
                - olens (torch.Tensor): Output lengths, shape (batch,).
                - hs_asr (torch.Tensor, optional): Last hidden state if return_hs is True.

        Note:
            This method supports multi-modal decoding by incorporating both text and speech
            inputs when available. The speech attention mechanism is used only if
            use_speech_attn is True and speech input is provided.
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        if speech is not None:
            speech_mask = (~make_pad_mask(speech_lens, maxlen=speech.size(1)))[
                :, None, :
            ].to(speech.device)
        else:
            speech_mask = None

        x = self.embed(tgt)
        if self.use_speech_attn:
            x, tgt_mask, memory, memory_mask, _, speech, speech_mask = self.decoders(
                x, tgt_mask, memory, memory_mask, None, speech, speech_mask
            )
        else:
            x, tgt_mask, memory, memory_mask = self.decoders(
                x, tgt_mask, memory, memory_mask
            )
        if self.normalize_before:
            x = self.after_norm(x)
            if return_hs:
                hs_asr = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)

        if return_hs:
            return x, olens, hs_asr

        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        speech: torch.Tensor = None,
        speech_mask: torch.Tensor = None,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
                Perform one step of the decoder forward pass.

        This method is designed for incremental decoding, processing one token at a time
        and optionally incorporating speech input.

        Args:
            tgt (torch.Tensor): Input token ids, shape (batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask, shape (batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2-, dtype=torch.bool in PyTorch 1.2+ (including 1.2).
            memory (torch.Tensor): Encoded memory, shape (batch, maxlen_in, feat).
            memory_mask (torch.Tensor, optional): Encoded memory mask, shape (batch, 1, maxlen_in).
            speech (torch.Tensor, optional): Speech input tensor, shape (batch, speech_maxlen, speech_feat).
            speech_mask (torch.Tensor, optional): Speech input mask, shape (batch, 1, speech_maxlen).
            cache (List[torch.Tensor], optional): Cached output list of shape (batch, max_time_out-1, size).
            return_hs (bool, optional): Whether to return the hidden state. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - y (torch.Tensor): Output tensor of shape (batch, maxlen_out, token)
                    if use_output_layer is True, else the last hidden state.
                - h_asr (torch.Tensor, optional): Hidden state if return_hs is True.
                - new_cache (List[torch.Tensor]): Updated cache for each decoder layer.

        Note:
            This method supports multi-modal decoding by incorporating both text and speech
            inputs when available. The speech attention mechanism is used only if
            use_speech_attn is True and speech input is provided.
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            if self.use_speech_attn:
                x, tgt_mask, memory, memory_mask, _, speech, speech_mask = decoder(
                    x,
                    tgt_mask,
                    memory,
                    memory_mask,
                    cache=c,
                    pre_memory=speech,
                    pre_memory_mask=speech_mask,
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, memory_mask, cache=c
                )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if return_hs:
            h_asr = y

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return y, h_asr, new_cache
        return y, new_cache

    def score(self, ys, state, x, speech=None):
        """
                Score a sequence of tokens.

        This method computes the log probability score for a given sequence of tokens,
        optionally incorporating speech input.

        Args:
            ys (torch.Tensor): Sequence of tokens to score, shape (sequence_length,).
            state (List[Any]): Previous decoder state.
            x (torch.Tensor): Encoder output, shape (1, encoder_output_size).
            speech (torch.Tensor, optional): Speech input tensor, shape (1, speech_maxlen, speech_feat).

        Returns:
            tuple: A tuple containing:
                - logp (torch.Tensor): Log probability scores for the input sequence,
                    shape (sequence_length, vocab_size).
                - state (List[Any]): Updated decoder state.

        Note:
            This method supports multi-modal scoring by incorporating both text and speech
            inputs when available. The speech attention mechanism is used only if
            use_speech_attn is True and speech input is provided. It is typically used
            in beam search and other decoding algorithms to evaluate candidate sequences.
        """
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            speech=speech.unsqueeze(0) if speech is not None else None,
            cache=state,
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        speech: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
                Score a batch of token sequences.

        This method computes the log probability scores for a batch of token sequences,
        optionally incorporating speech input for multi-modal scoring.

        Args:
            ys (torch.Tensor): Batch of token sequences, shape (batch_size, sequence_length).
            states (List[Any]): List of scorer states for prefix tokens.
            xs (torch.Tensor): Batch of encoder outputs, shape (batch_size, max_length, encoder_output_size).
            speech (torch.Tensor, optional): Batch of speech inputs, shape (batch_size, speech_maxlen, speech_feat).

        Returns:
            tuple: A tuple containing:
                - logp (torch.Tensor): Batch of log probability scores for the next token,
                    shape (batch_size, vocab_size).
                - state_list (List[List[Any]]): Updated list of scorer states for each sequence in the batch.

        Note:
            This method is optimized for batch processing, which is more efficient than
            scoring sequences individually. It supports multi-modal scoring by incorporating
            both text and speech inputs when available. The speech attention mechanism is used
            only if use_speech_attn is True and speech input is provided. This method is
            particularly useful for beam search or batch decoding in multi-modal tasks.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(
            ys, ys_mask, xs, speech=speech, cache=batch_state
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
