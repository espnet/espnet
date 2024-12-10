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


class BaseTransformerDecoder(
    AbsDecoder, BatchScorerInterface, MaskParallelScorerInterface
):
    """
    Base class of Transformer decoder module.

    This class implements a base structure for a Transformer decoder used in
    automatic speech recognition (ASR) tasks. It defines the main architecture,
    input handling, and forward propagation methods required for decoding.

    Attributes:
        embed (torch.nn.Sequential): Input embedding layer, which can be an 
            embedding layer followed by positional encoding or a linear layer.
        after_norm (LayerNorm): Layer normalization applied before the first 
            decoder block if `normalize_before` is set to True.
        output_layer (torch.nn.Linear): Output layer that maps decoder outputs 
            to the vocabulary size, if `use_output_layer` is True.
        _output_size_bf_softmax (int): Dimension of the output before applying 
            softmax, set to the attention dimension.
        decoders (List[DecoderLayer]): List of decoder layers (to be set by 
            inheritance).
        batch_ids (torch.Tensor): Tensor containing batch IDs for processing.

    Args:
        vocab_size (int): The size of the output vocabulary.
        encoder_output_size (int): The dimension of the encoder's output.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. 
            Default is 0.1.
        input_layer (str, optional): Type of input layer ('embed' or 'linear'). Default is 'embed'.
        use_output_layer (bool, optional): Whether to use an output layer. Default is True.
        pos_enc_class (Type[PositionalEncoding], optional): Class for positional encoding. 
            Default is PositionalEncoding.
        normalize_before (bool, optional): Whether to apply layer normalization before the 
            first decoder block. Default is True.

    Examples:
        # Initialize the decoder
        decoder = BaseTransformerDecoder(
            vocab_size=5000,
            encoder_output_size=256,
            dropout_rate=0.1,
            input_layer='embed'
        )

        # Forward pass
        output, lengths = decoder(
            hs_pad=encoded_memory_tensor,
            hlens=encoded_memory_lengths,
            ys_in_pad=input_token_ids,
            ys_in_lens=input_lengths
        )

    Raises:
        ValueError: If `input_layer` is not 'embed' or 'linear'.

    Note:
        This class should be inherited by specific transformer decoder 
        implementations to provide concrete decoder layer configurations.
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
        Forward decoder.

        This method takes the encoded memory and input token IDs to produce the
        output token scores before softmax. It can also return hidden states
        if specified.

        Args:
            hs_pad (torch.Tensor): Encoded memory with shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of the encoded sequences with shape (batch).
            ys_in_pad (torch.Tensor): Input token IDs with shape (batch, maxlen_out).
                If `input_layer` is "embed", it represents token IDs; otherwise, 
                it should be a tensor of shape (batch, maxlen_out, #mels).
            ys_in_lens (torch.Tensor): Lengths of the input sequences with shape (batch).
            return_hs (bool, optional): Whether to return the last hidden output
                before the output layer. Defaults to False.
            return_all_hs (bool, optional): Whether to return all hidden 
                intermediate states. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax with shape
                (batch, maxlen_out, token) if `use_output_layer` is True.
                - olens (torch.Tensor): Lengths of the output sequences with shape (batch,).

        Examples:
            >>> hs_pad = torch.randn(2, 10, 512)  # Example memory
            >>> hlens = torch.tensor([10, 8])  # Lengths of memory
            >>> ys_in_pad = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Input token IDs
            >>> ys_in_lens = torch.tensor([3, 3])  # Lengths of input sequences
            >>> output_scores, output_lengths = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)

        Note:
            Ensure that the input tensors are appropriately padded and have the 
            correct dimensions as described in the arguments.

        Raises:
            ValueError: If the shapes of input tensors do not match the expected 
            dimensions or if any input is malformed.
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
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
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
        Compute the score for a given input sequence.

        This method performs a single step of scoring for the input 
        sequence `ys` based on the provided state and encoder output `x`. 
        It can optionally return the hidden state if requested.

        Args:
            ys (torch.Tensor): Input token ids, shape (maxlen_out), 
                of type int64.
            state (List[torch.Tensor]): Cached states for the decoder 
                from previous steps.
            x (torch.Tensor): Encoded memory, shape (1, maxlen_in, feat), 
                of type float32.
            return_hs (bool, optional): If True, the hidden state 
                corresponding to the input tokens will be returned. 
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - logp (torch.Tensor): Log probabilities of the next token, 
                  shape (vocab_size).
                - state (List[torch.Tensor]): Updated state after scoring.

        Examples:
            >>> decoder = TransformerDecoder(vocab_size=5000, 
            ... encoder_output_size=256)
            >>> ys = torch.tensor([1, 2, 3])  # Example token ids
            >>> state = [None]  # Initial state
            >>> x = torch.randn(1, 10, 256)  # Example encoder output
            >>> logp, new_state = decoder.score(ys, state, x)

        Note:
            This method is typically used in the decoding process to 
            compute the next token's probability given the previous 
            tokens and the encoder output.
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
        Score new token batch.

        This method computes the scores for the next token based on the input 
        tokens, the current states, and the encoder features. It can return 
        hidden states if requested.

        Args:
            ys (torch.Tensor): A tensor of shape (n_batch, ylen) containing 
                the prefix tokens in int64 format.
            states (List[Any]): A list of states for the prefix tokens, where 
                each state corresponds to the current state of the decoder.
            xs (torch.Tensor): A tensor of shape (n_batch, xlen, n_feat) 
                representing the encoder features that generated the prefix 
                tokens.

            return_hs (bool, optional): If True, the method will return the 
                hidden states along with the scores. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[Any]]: A tuple containing:
                - A tensor of shape (n_batch, n_vocab) representing the 
                  scores for the next token.
                - A list of next state lists for the prefix tokens.

        Examples:
            >>> decoder = TransformerDecoder(vocab_size=100, encoder_output_size=256)
            >>> ys = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> states = [None, None]
            >>> xs = torch.rand(2, 10, 256)
            >>> scores, next_states = decoder.batch_score(ys, states, xs)
            >>> print(scores.shape)  # Should print: torch.Size([2, 100])

        Note:
            Ensure that the input tensors are properly shaped and that the 
            states list matches the expected format for the decoder.

        Raises:
            ValueError: If the dimensions of the input tensors do not match 
            the expected shapes.
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
        """
        Forward one step in a partially autoregressive manner.

        This method processes the input tokens and computes the output scores
        for the next tokens in a partially autoregressive manner, allowing for
        efficient decoding during beam search or similar scenarios.

        Args:
            tgt: Input token ids, int64 of shape 
                (n_mask * n_beam, maxlen_out).
            tgt_mask: Input token mask of shape 
                (n_mask * n_beam, maxlen_out).
                The data type should be torch.uint8 for PyTorch versions < 1.2
                and torch.bool for PyTorch versions >= 1.2.
            tgt_lengths: Lengths of the input sequences, shape (n_mask * n_beam,).
            memory: Encoded memory from the encoder, float32 of shape 
                (batch, maxlen_in, feat).
            cache: Cached output list for each decoder layer, which can be 
                used to store previous hidden states and facilitate efficient 
                decoding.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - y: Output token scores, float32 of shape 
                  (n_mask * n_beam, maxlen_out, vocab_size).
                - cache: Updated cache containing hidden states for each 
                  decoder layer.

        Examples:
            >>> decoder = TransformerDecoder(vocab_size=1000, 
            ... encoder_output_size=512)
            >>> tgt = torch.randint(0, 1000, (2, 10))  # 2 sequences of length 10
            >>> tgt_mask = torch.ones((2, 10), dtype=torch.bool)
            >>> tgt_lengths = torch.tensor([10, 10])
            >>> memory = torch.randn(2, 15, 512)  # 2 sequences of length 15
            >>> output, updated_cache = decoder.forward_partially_AR(
            ...     tgt, tgt_mask, tgt_lengths, memory
            ... )

        Note:
            This method is particularly useful in scenarios where decoding
            needs to be efficient, such as during beam search or when 
            generating sequences with partial context.
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
        """
        Score a batch of new tokens in a partially autoregressive manner.

        This method evaluates a batch of token sequences and returns their scores
        along with the updated states. It is specifically designed for use in 
        scenarios where the sequences are generated in a partially autoregressive 
        manner.

        Args:
            ys (torch.Tensor): Tensor of shape (n_mask * n_beam, ylen) containing 
                the token sequences for which scores are to be computed. Each 
                element is an integer representing a token ID.
            states (List[Any]): A list of states for the scorer, where each state 
                corresponds to a prefix of tokens in `ys`.
            xs (torch.Tensor): Tensor of shape (n_batch, xlen, n_feat) representing 
                the encoder features that are used to generate the scores for 
                the sequences in `ys`.
            yseq_lengths (torch.Tensor): Tensor of shape (n_mask * n_beam,) 
                containing the lengths of the sequences in `ys`, used to create 
                the appropriate attention masks.

        Returns:
            Tuple[torch.Tensor, List[Any]]: A tuple where the first element is 
            a tensor of shape (n_mask * n_beam, n_vocab) containing the 
            scores for the next token in the sequence, and the second element 
            is a list of updated states for each sequence in `ys`.

        Examples:
            >>> ys = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> states = [None, None]
            >>> xs = torch.randn(2, 10, 256)  # Example encoder features
            >>> yseq_lengths = torch.tensor([3, 3])
            >>> scores, updated_states = batch_score_partially_AR(ys, states, xs, 
            ...                                                  yseq_lengths)

        Note:
            This method requires that the lengths in `yseq_lengths` are consistent 
            with the sequences in `ys`. It also assumes that the appropriate 
            padding has been applied to `ys` and `xs`.
        """
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
    Transformer Decoder for sequence-to-sequence tasks.

    This class implements a Transformer decoder architecture, which is 
    designed to work in conjunction with a Transformer encoder. The decoder 
    generates sequences based on the encoded representations from the 
    encoder, using mechanisms such as multi-head attention and feed-forward 
    networks.

    Args:
        vocab_size (int): The size of the vocabulary, representing the number 
            of unique tokens in the output.
        encoder_output_size (int): The dimension of the output from the 
            encoder, which the decoder will attend to.
        attention_heads (int, optional): The number of attention heads to use 
            in the multi-head attention mechanism. Default is 4.
        linear_units (int, optional): The number of units in the position-wise 
            feed-forward layer. Default is 2048.
        num_blocks (int, optional): The number of decoder blocks (layers) to 
            stack. Default is 6.
        dropout_rate (float, optional): The dropout rate for regularization. 
            Default is 0.1.
        positional_dropout_rate (float, optional): The dropout rate for 
            positional encoding. Default is 0.1.
        self_attention_dropout_rate (float, optional): The dropout rate 
            applied to the self-attention mechanism. Default is 0.0.
        src_attention_dropout_rate (float, optional): The dropout rate applied 
            to the source attention mechanism. Default is 0.0.
        input_layer (str, optional): The type of input layer to use; either 
            'embed' for embedding layer or 'linear' for a linear layer. 
            Default is 'embed'.
        use_output_layer (bool, optional): Whether to use an output layer 
            for final token scoring. Default is True.
        pos_enc_class (type, optional): The class to use for positional 
            encoding, e.g., PositionalEncoding or ScaledPositionalEncoding. 
            Default is PositionalEncoding.
        normalize_before (bool, optional): Whether to apply layer normalization 
            before the first decoder block. Default is True.
        concat_after (bool, optional): Whether to concatenate the input and 
            output of the attention layer before applying an additional 
            linear layer. Default is False.
        layer_drop_rate (float, optional): The dropout rate for layer 
            dropping. Default is 0.0.
        qk_norm (bool, optional): Whether to apply normalization to the 
            query-key dot product in attention. Default is False.
        use_flash_attn (bool, optional): Whether to use flash attention for 
            improved performance. Default is True.

    Examples:
        >>> decoder = TransformerDecoder(
        ...     vocab_size=5000,
        ...     encoder_output_size=512,
        ...     attention_heads=8,
        ...     linear_units=2048,
        ...     num_blocks=6,
        ... )
        >>> hs_pad = torch.randn(32, 10, 512)  # Batch of 32, 10 time steps
        >>> hlens = torch.tensor([10] * 32)  # All sequences of length 10
        >>> ys_in_pad = torch.randint(0, 5000, (32, 15))  # Batch of 32, 15 tokens
        >>> ys_in_lens = torch.tensor([15] * 32)  # All sequences of length 15
        >>> output, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        The decoder expects the encoder's output to be passed in as 
        `hs_pad`, along with the lengths of those sequences in `hlens`.
        The `ys_in_pad` contains the input tokens for the decoder, and 
        `ys_in_lens` provides the lengths of those input sequences.

    Raises:
        ValueError: If the `input_layer` argument is not 'embed' or 'linear'.
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
        qk_norm: bool = False,
        use_flash_attn: bool = True,
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

        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn
            except Exception:
                use_flash_attn = False

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    True,
                    False,
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    True,
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
    Lightweight Convolution Transformer Decoder.

    This class implements a transformer decoder that utilizes lightweight 
    convolution layers in its architecture. It is designed for tasks 
    such as automatic speech recognition (ASR) and can be used as a 
    part of larger neural network models.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        encoder_output_size (int): The output dimension of the encoder.
        attention_heads (int): The number of attention heads for multi-head 
            attention.
        linear_units (int): The number of units in the position-wise feed 
            forward layer.
        num_blocks (int): The number of decoder blocks in the architecture.
        dropout_rate (float): The dropout rate to apply to layers.
        positional_dropout_rate (float): The dropout rate for positional 
            encodings.
        self_attention_dropout_rate (float): The dropout rate for self 
            attention.
        src_attention_dropout_rate (float): The dropout rate for source 
            attention.
        input_layer (str): The type of input layer to use ('embed' or 
            'linear').
        use_output_layer (bool): Flag indicating whether to use an output 
            layer.
        pos_enc_class: The class used for positional encoding.
        normalize_before (bool): Whether to apply layer normalization before 
            the first block.
        concat_after (bool): Whether to concatenate the input and output of 
            the attention layer.
        conv_wshare (int): The number of shared weights for convolutional 
            layers.
        conv_kernel_length (Sequence[int]): A sequence specifying the kernel 
            length for each convolutional layer.
        conv_usebias (bool): Whether to use bias in convolutional layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        encoder_output_size (int): The output dimension of the encoder.
        attention_heads (int, optional): The number of attention heads. 
            Defaults to 4.
        linear_units (int, optional): The number of units in the 
            position-wise feed forward layer. Defaults to 2048.
        num_blocks (int, optional): The number of decoder blocks. 
            Defaults to 6.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): The dropout rate for 
            positional encodings. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): The dropout rate 
            for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): The dropout rate for 
            source attention. Defaults to 0.0.
        input_layer (str, optional): The type of input layer ('embed' or 
            'linear'). Defaults to 'embed'.
        use_output_layer (bool, optional): Flag indicating whether to use an 
            output layer. Defaults to True.
        pos_enc_class: The class used for positional encoding. Defaults to 
            PositionalEncoding.
        normalize_before (bool, optional): Whether to apply layer normalization 
            before the first block. Defaults to True.
        concat_after (bool, optional): Whether to concatenate the input and 
            output of the attention layer. Defaults to False.
        conv_wshare (int, optional): The number of shared weights for 
            convolutional layers. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): A sequence specifying 
            the kernel length for each convolutional layer. Defaults to 
            (11, 11, 11, 11, 11, 11).
        conv_usebias (bool, optional): Whether to use bias in convolutional 
            layers. Defaults to False.

    Raises:
        ValueError: If the length of `conv_kernel_length` does not match 
            `num_blocks`.

    Examples:
        >>> decoder = LightweightConvolutionTransformerDecoder(
        ...     vocab_size=5000,
        ...     encoder_output_size=256,
        ...     num_blocks=6,
        ...     conv_kernel_length=[3, 5, 7, 9, 11, 13]
        ... )
        >>> input_tensor = torch.randint(0, 5000, (32, 10))  # (batch, seq_len)
        >>> output, olens = decoder.forward(input_tensor, hlens=None, ys_in_pad=input_tensor, ys_in_lens=None)
    
    Note:
        This implementation is suitable for both training and inference 
        scenarios. The `forward` method is used to process the input data 
        through the decoder layers.
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
    Lightweight Convolution 2D Transformer Decoder.

    This class implements a transformer decoder that utilizes lightweight 
    2D convolutions in its architecture. It inherits from the 
    BaseTransformerDecoder class and is designed to facilitate 
    sequence-to-sequence tasks, such as automatic speech recognition.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimension of the encoder's output.
        attention_heads (int): Number of attention heads in multi-head 
            attention.
        linear_units (int): Number of units in position-wise feed forward 
            networks.
        num_blocks (int): Number of decoder blocks.
        dropout_rate (float): Dropout rate applied in various layers.
        positional_dropout_rate (float): Dropout rate for positional encoding.
        self_attention_dropout_rate (float): Dropout rate for self-attention.
        src_attention_dropout_rate (float): Dropout rate for source 
            attention.
        input_layer (str): Type of input layer ('embed' or 'linear').
        use_output_layer (bool): Flag to indicate if output layer is used.
        pos_enc_class: Class used for positional encoding.
        normalize_before (bool): Flag to indicate if normalization is applied 
            before the first block.
        concat_after (bool): Flag to indicate if concatenation is applied 
            after attention.
        conv_wshare (int): Number of shared weights for convolution.
        conv_kernel_length (Sequence[int]): Lengths of convolution kernels 
            for each block.
        conv_usebias (bool): Flag to indicate if bias is used in convolutions.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimension of the encoder's output.
        attention_heads (int, optional): Number of attention heads. Default is 4.
        linear_units (int, optional): Number of units in position-wise feed 
            forward networks. Default is 2048.
        num_blocks (int, optional): Number of decoder blocks. Default is 6.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
        positional_dropout_rate (float, optional): Dropout rate for 
            positional encoding. Default is 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for 
            self-attention. Default is 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for 
            source attention. Default is 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 'linear').
            Default is 'embed'.
        use_output_layer (bool, optional): Flag to indicate if output layer 
            is used. Default is True.
        pos_enc_class: Class used for positional encoding. Default is 
            PositionalEncoding.
        normalize_before (bool, optional): Flag to indicate if normalization 
            is applied before the first block. Default is True.
        concat_after (bool, optional): Flag to indicate if concatenation is 
            applied after attention. Default is False.
        conv_wshare (int, optional): Number of shared weights for convolution. 
            Default is 4.
        conv_kernel_length (Sequence[int], optional): Lengths of convolution 
            kernels for each block. Default is (11, 11, 11, 11, 11, 11).
        conv_usebias (bool, optional): Flag to indicate if bias is used in 
            convolutions. Default is False.

    Raises:
        ValueError: If the length of conv_kernel_length does not match 
            num_blocks.

    Examples:
        >>> decoder = LightweightConvolution2DTransformerDecoder(
        ...     vocab_size=1000,
        ...     encoder_output_size=512,
        ...     num_blocks=6,
        ...     conv_kernel_length=[3, 5, 7, 9, 11, 13]
        ... )
        >>> hs_pad = torch.rand(32, 50, 512)  # Batch of encoded memory
        >>> hlens = torch.tensor([50] * 32)  # Lengths of the input
        >>> ys_in_pad = torch.randint(0, 1000, (32, 20))  # Input tokens
        >>> ys_in_lens = torch.tensor([20] * 32)  # Lengths of the output
        >>> output, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)
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
    Dynamic Convolution Transformer Decoder for sequence generation.

    This class implements a dynamic convolution mechanism in the Transformer
    decoder architecture, allowing for adaptive convolutional operations
    within the decoder layers. It is designed to facilitate improved 
    performance in tasks such as automatic speech recognition (ASR).

    Attributes:
        vocab_size (int): Size of the vocabulary for output tokens.
        encoder_output_size (int): Size of the encoder's output features.
        attention_heads (int): Number of attention heads for multi-head 
            attention.
        linear_units (int): Number of units in the position-wise feed 
            forward layer.
        num_blocks (int): Number of decoder blocks to stack.
        dropout_rate (float): Dropout rate for regularization.
        positional_dropout_rate (float): Dropout rate for positional encoding.
        self_attention_dropout_rate (float): Dropout rate for self-attention.
        src_attention_dropout_rate (float): Dropout rate for source attention.
        input_layer (str): Type of input layer ('embed' or 'linear').
        use_output_layer (bool): Flag to determine if an output layer is used.
        pos_enc_class: Class for positional encoding.
        normalize_before (bool): Flag for normalization before the first block.
        concat_after (bool): Flag for concatenation of attention inputs and 
            outputs.
        conv_wshare (int): Number of shared weights for dynamic convolution.
        conv_kernel_length (Sequence[int]): Length of the convolution kernels 
            for each block.
        conv_usebias (bool): Flag to use bias in convolution operations.

    Args:
        vocab_size (int): Size of the vocabulary for output tokens.
        encoder_output_size (int): Size of the encoder's output features.
        attention_heads (int, optional): Number of attention heads for 
            multi-head attention. Defaults to 4.
        linear_units (int, optional): Number of units in the position-wise 
            feed forward layer. Defaults to 2048.
        num_blocks (int, optional): Number of decoder blocks to stack. 
            Defaults to 6.
        dropout_rate (float, optional): Dropout rate for regularization. 
            Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional 
            encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for 
            self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for 
            source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer ('embed' or 
            'linear'). Defaults to 'embed'.
        use_output_layer (bool, optional): Flag to determine if an output 
            layer is used. Defaults to True.
        pos_enc_class: Class for positional encoding. Defaults to 
            PositionalEncoding.
        normalize_before (bool, optional): Flag for normalization before the 
            first block. Defaults to True.
        concat_after (bool, optional): Flag for concatenation of attention 
            inputs and outputs. Defaults to False.
        conv_wshare (int, optional): Number of shared weights for dynamic 
            convolution. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): Length of the 
            convolution kernels for each block. Defaults to (11, 11, 11, 
            11, 11, 11).
        conv_usebias (bool, optional): Flag to use bias in convolution 
            operations. Defaults to False.

    Raises:
        ValueError: If `conv_kernel_length` does not match the number of 
            blocks.

    Examples:
        >>> decoder = DynamicConvolutionTransformerDecoder(
        ...     vocab_size=1000,
        ...     encoder_output_size=256,
        ...     num_blocks=6,
        ...     conv_kernel_length=(3, 5, 7, 9, 11, 13)
        ... )
        >>> hs_pad = torch.randn(32, 50, 256)  # Example encoded memory
        >>> hlens = torch.tensor([50] * 32)  # Lengths of encoded memory
        >>> ys_in_pad = torch.randint(0, 1000, (32, 30))  # Input tokens
        >>> ys_in_lens = torch.tensor([30] * 32)  # Lengths of input tokens
        >>> output, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        The dynamic convolution allows for varying kernel sizes and shared 
        weights across different decoder layers, enhancing model flexibility 
        and capacity.

    Todo:
        - Implement additional features for attention mechanisms.
        - Explore integration with other types of layers and modules.
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
    Dynamic Convolution 2D Transformer Decoder.

    This class implements a transformer decoder that utilizes dynamic
    convolution with 2D kernels to enhance the attention mechanism.
    It is suitable for sequence-to-sequence tasks, particularly in
    automatic speech recognition (ASR).

    Args:
        vocab_size (int): The size of the vocabulary (number of tokens).
        encoder_output_size (int): The output dimension from the encoder.
        attention_heads (int, optional): The number of attention heads.
            Defaults to 4.
        linear_units (int, optional): The number of units in the 
            position-wise feed-forward layer. Defaults to 2048.
        num_blocks (int, optional): The number of decoder blocks.
            Defaults to 6.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): The dropout rate for
            positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): The dropout rate
            for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): The dropout rate
            for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer, either 'embed' 
            or 'linear'. Defaults to 'embed'.
        use_output_layer (bool, optional): Whether to use an output layer.
            Defaults to True.
        pos_enc_class: The class used for positional encoding. Defaults to
            PositionalEncoding.
        normalize_before (bool, optional): Whether to apply layer normalization
            before the first block. Defaults to True.
        concat_after (bool, optional): Whether to concatenate the input and 
            output of the attention layer. Defaults to False.
        conv_wshare (int, optional): The number of shared weights for 
            convolution. Defaults to 4.
        conv_kernel_length (Sequence[int], optional): The lengths of the 
            convolution kernels for each block. Defaults to (11, 11, 11, 
            11, 11, 11).
        conv_usebias (bool, optional): Whether to use bias in the 
            convolution layers. Defaults to False.

    Raises:
        ValueError: If the length of `conv_kernel_length` does not match 
        `num_blocks`.

    Examples:
        >>> decoder = DynamicConvolution2DTransformerDecoder(
        ...     vocab_size=1000,
        ...     encoder_output_size=256,
        ...     num_blocks=6,
        ...     conv_kernel_length=(3, 5, 7, 9, 11, 13)
        ... )
        >>> hs_pad = torch.randn(32, 50, 256)  # (batch, maxlen_in, feat)
        >>> hlens = torch.randint(1, 51, (32,))  # (batch)
        >>> ys_in_pad = torch.randint(0, 1000, (32, 20))  # (batch, maxlen_out)
        >>> ys_in_lens = torch.randint(1, 21, (32,))  # (batch)
        >>> output, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        This decoder is designed to work with the corresponding encoder
        that provides the necessary context through the `hs_pad` input.
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
    Transformer decoder with multi-dimensional attention.

    This class implements a Transformer decoder that integrates 
    multi-dimensional attention capabilities. It extends the 
    BaseTransformerDecoder class and allows the incorporation 
    of speech attention, enabling enhanced performance in 
    speech-related tasks.

    Attributes:
        use_speech_attn (bool): Flag indicating whether to use 
            speech attention in the decoding process.

    Args:
        vocab_size (int): Size of the vocabulary for output tokens.
        encoder_output_size (int): Dimensionality of the encoder's 
            output.
        attention_heads (int): Number of attention heads for multi-head 
            attention (default: 4).
        linear_units (int): Number of units in the position-wise feed 
            forward layer (default: 2048).
        num_blocks (int): Number of decoder blocks (default: 6).
        dropout_rate (float): Dropout rate applied to the layers 
            (default: 0.1).
        positional_dropout_rate (float): Dropout rate applied to 
            positional encoding (default: 0.1).
        self_attention_dropout_rate (float): Dropout rate for self 
            attention (default: 0.0).
        src_attention_dropout_rate (float): Dropout rate for source 
            attention (default: 0.0).
        input_layer (str): Type of input layer; either "embed" or 
            "linear" (default: "embed").
        use_output_layer (bool): Flag indicating whether to use an 
            output layer (default: True).
        pos_enc_class: Class for positional encoding (default: 
            PositionalEncoding).
        normalize_before (bool): Flag indicating whether to apply 
            layer normalization before the first block (default: True).
        concat_after (bool): Flag indicating whether to concatenate 
            attention layer's input and output (default: False).
        use_speech_attn (bool): Flag indicating whether to use speech 
            attention (default: True).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the decoded 
        token scores before softmax and the lengths of the output 
        sequences.

    Examples:
        >>> decoder = TransformerMDDecoder(vocab_size=1000, 
        ...                                  encoder_output_size=512)
        >>> hs_pad = torch.randn(32, 10, 512)  # (batch, maxlen_in, feat)
        >>> hlens = torch.tensor([10] * 32)  # (batch)
        >>> ys_in_pad = torch.randint(0, 1000, (32, 20))  # (batch, maxlen_out)
        >>> ys_in_lens = torch.tensor([20] * 32)  # (batch)
        >>> output, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        This decoder is particularly useful for tasks that involve 
        processing both text and speech inputs, enabling better 
        context understanding and more accurate outputs.

    Raises:
        ValueError: If the `input_layer` argument is not one of 
            "embed" or "linear".
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

        This method performs the forward pass of the TransformerMDDecoder, 
        processing the encoded memory and input token sequences to produce 
        decoded token scores. It can also utilize speech features if provided.

        Args:
            hs_pad (torch.Tensor): Encoded memory, shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of the encoder outputs, shape (batch).
            ys_in_pad (torch.Tensor): Input token IDs, shape (batch, maxlen_out).
                If `input_layer` is "embed", otherwise, input tensor shape is 
                (batch, maxlen_out, #mels).
            ys_in_lens (torch.Tensor): Lengths of the input sequences, shape (batch).
            speech (torch.Tensor, optional): Encoded speech features, 
                shape (batch, maxlen_in, feat). Defaults to None.
            speech_lens (torch.Tensor, optional): Lengths of the speech 
                sequences, shape (batch). Defaults to None.
            return_hs (bool, optional): If True, return the last hidden 
                state corresponding to the output. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax, 
                  shape (batch, maxlen_out, vocab_size) if 
                  `use_output_layer` is True.
                - olens (torch.Tensor): Lengths of the output sequences, 
                  shape (batch,).
        
        Examples:
            >>> hs_pad = torch.randn(32, 50, 256)  # Example memory tensor
            >>> hlens = torch.randint(1, 51, (32,))
            >>> ys_in_pad = torch.randint(0, 100, (32, 20))  # Example input IDs
            >>> ys_in_lens = torch.randint(1, 21, (32,))
            >>> decoder = TransformerMDDecoder(100, 256)
            >>> output, output_lengths = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

        Note:
            If `speech` is provided, the decoder will leverage the speech 
            attention mechanism for enhanced decoding performance.
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
        Forward one step through the decoder.

        This method performs a single step of decoding, where the model
        generates the next token based on the current input tokens, the 
        encoded memory from the encoder, and optionally, speech features 
        and their masks. The output is the predicted token and updated cache 
        for future decoding steps.

        Args:
            tgt: Input token IDs, of shape (batch, maxlen_out), 
                 where each entry is an integer representing a token.
            tgt_mask: Input token mask of shape (batch, maxlen_out).
                      It can be of dtype torch.uint8 in PyTorch 1.2- 
                      or dtype torch.bool in PyTorch 1.2+.
            memory: Encoded memory from the encoder, of shape 
                    (batch, maxlen_in, feat).
            memory_mask: (Optional) Mask for the memory, of shape 
                         (batch, 1, maxlen_in).
            speech: (Optional) Encoded speech features, of shape 
                    (batch, maxlen_in, feat).
            speech_mask: (Optional) Mask for the speech, of shape 
                         (batch, 1, maxlen_in).
            cache: (Optional) Cached output list from previous steps, 
                   of shape (batch, max_time_out-1, size).
            return_hs: Whether to return the hidden state corresponding 
                        to the input tokens, useful for debugging or 
                        further processing.

        Returns:
            A tuple containing:
                - y: Output token scores, of shape (batch, maxlen_out, token).
                     This is the log probabilities of the next token.
                - cache: Updated cache for future decoding steps.
        
        Examples:
            >>> tgt = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> tgt_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
            >>> memory = torch.rand(2, 10, 512)  # Random memory
            >>> output, new_cache = decoder.forward_one_step(tgt, tgt_mask, memory)
        
        Note:
            Ensure that the input tensor dimensions are compatible with 
            the model's expectations. The `speech` and `speech_mask` 
            parameters are optional and should only be provided if 
            speech features are being used in the decoding process.
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
        Calculate the score for a given sequence and update the state.

        This method computes the log probability of the next token in the 
        sequence based on the provided input features and updates the 
        internal state of the decoder.

        Args:
            ys (torch.Tensor): The input token IDs of shape (n_tokens,).
            state (List[torch.Tensor]): The cached state of the decoder.
            x (torch.Tensor): The encoder output features of shape 
                (1, input_length, feature_dim).
            speech (torch.Tensor, optional): The encoded speech features of 
                shape (1, input_length, feature_dim). Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - logp (torch.Tensor): The log probability of the next token 
                  of shape (vocab_size,).
                - state (List[torch.Tensor]): The updated state of the 
                  decoder.

        Examples:
            >>> ys = torch.tensor([1, 2, 3])  # Example token IDs
            >>> state = [None]  # Initial state
            >>> x = torch.randn(1, 10, 512)  # Example encoder output
            >>> logp, updated_state = decoder.score(ys, state, x)

        Note:
            The method can handle optional speech features if provided. 
            This is useful for tasks that involve speech input.
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
        Score new token batch.

        Args:
            ys (torch.Tensor): A tensor of shape (n_batch, ylen) containing 
                the prefix tokens as int64.
            states (List[Any]): A list of scorer states corresponding to 
                the prefix tokens.
            xs (torch.Tensor): A tensor of shape (n_batch, xlen, n_feat) 
                representing the encoder features that generate `ys`.
            speech (torch.Tensor, optional): A tensor of shape (n_batch, 
                s_len, n_feat) representing the encoded speech features. 
                Defaults to None.

        Returns:
            tuple[torch.Tensor, List[Any]]: A tuple containing:
                - A tensor of shape (n_batch, n_vocab) representing the 
                  batchified scores for the next token.
                - A list of next state lists for `ys`.

        Examples:
            >>> decoder = TransformerMDDecoder(vocab_size=5000, encoder_output_size=256)
            >>> ys = torch.randint(0, 5000, (2, 10))  # Example input
            >>> states = [None, None]  # Example states
            >>> xs = torch.rand(2, 15, 256)  # Example encoder features
            >>> logp, next_states = decoder.batch_score(ys, states, xs)
            >>> print(logp.shape)  # Output: torch.Size([2, 5000])
        
        Note:
            This method performs batch scoring for the next tokens 
            given the input sequences and their corresponding states.
            It supports optional speech features to be used during scoring.
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
