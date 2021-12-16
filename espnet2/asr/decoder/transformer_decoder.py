# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import (
    DecoderLayer as TransDecoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.utils import check_state
from espnet.nets.pytorch_backend.transducer.utils import pad_batch_state
from espnet.nets.pytorch_backend.transducer.utils import pad_sequence
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
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import Hypothesis


class BaseTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        use_attention: whether to use second self-attn module (transducer)
        embed_pad: embedding idx for transducer model (eq. to blank)

    """

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
        use_attention: bool = True,
        embed_pad: Optional[int] = None,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim, padding_idx=embed_pad),
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

        self.blank = embed_pad
        self.dunits = attention_dim
        self.odim = vocab_size

        self.use_attention = use_attention
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        # Must set by the inheritance
        self.decoders = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        x = self.embed(tgt)

        if self.use_attention:
            # tgt_mask: (B, 1, L)
            tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
            # m: (1, L, L)
            m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
            # tgt_mask: (B, L, L)
            tgt_mask = tgt_mask & m

            memory = hs_pad
            memory_mask = (~make_pad_mask(hlens))[:, None, :].to(memory.device)

            x, tgt_mask, memory, memory_mask = self.decoders(
                x, tgt_mask, memory, memory_mask
            )
            olens = tgt_mask.sum(1)
        else:
            tgt_mask = tgt != self.blank
            m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
            tgt_mask = tgt_mask.unsqueeze(-2) & m

            x, tgt_mask = self.decoders(x, tgt_mask)
            olens = tgt_mask

        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        return x, olens

    def init_state(self, init_tensor: torch.Tensor = None) -> Optional[List]:
        """Initialize decoder states."""
        if self.blank == 0:
            state = [None] * len(self.decoders)
        else:
            state = None

        return state

    def init_batch_states(self, init_tensor: torch.Tensor = None) -> Optional[List]:
        """Initialize decoder states."""

        return self.init_state()

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
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
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

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
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def step_transducer(
        self, hyp: Hypothesis, cache: dict, init_tensor: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """Forward one step.

        Args:
            hyp: Hypothesis
            cache: States cache

        Returns:
            y: Decoder outputs (1, D_dec)
            new_state: Decoder outputs [L x (1, max_len, D_dec)]
            lm_tokens: Token id for LM (1)

        """
        tgt = to_device(self, torch.tensor(hyp.yseq).unsqueeze(0))
        lm_tokens = tgt[:, -1]

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, new_state = cache[str_yseq]
        else:
            tgt_mask = to_device(self, subsequent_mask(len(hyp.yseq)).unsqueeze(0))

            state = check_state(hyp.dec_state, (tgt.size(1) - 1), self.blank)

            tgt = self.embed(tgt)

            new_state = []
            for s, decoder in zip(state, self.decoders):
                tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
                new_state.append(tgt)

            if self.normalize_before:
                y = self.after_norm(tgt[:, -1])
            else:
                y = tgt[:, -1]

            cache[str_yseq] = (y, new_state)

        return y, new_state, lm_tokens

    def batch_step_transducer(
        self,
        hyps: List,
        batch_states: List[torch.Tensor],
        cache: dict,
        init_tensor: torch.Tensor = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward batch one step.

        Args:
            hyps: Batch of hypotheses
            batch_states: Decoder states [L x (B, max_len, D_dim)]
            cache: States cache

        Returns:
            batch_y: Decoder outputs (B, D_dec)
            batch_states: Decoder states [L x (B, max_len, dec_dim)]
            lm_tokens: Batch of token ids for LM (B, 1)

        """
        final_batch = len(hyps)

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hyps):
            str_yseq = "".join([str(x) for x in hyp.yseq])

            if str_yseq in cache:
                done[i] = (*cache[str_yseq], hyp.yseq)
            else:
                tokens.append(hyp.yseq)
                process.append((str_yseq, hyp.dec_state, hyp.yseq))

        if process:
            batch = len(tokens)

            tokens = pad_sequence(tokens, self.blank)
            b_tokens = to_device(self, torch.LongTensor(tokens).view(batch, -1))

            tgt_mask = to_device(
                self,
                subsequent_mask(b_tokens.size(-1)).unsqueeze(0).expand(batch, -1, -1),
            )

            dec_state = self.init_batch_states()

            dec_state = self._create_batch_states(
                dec_state,
                [p[1] for p in process],
                tokens,
            )

            tgt = self.embed(b_tokens)

            next_state = []
            for s, decoder in zip(dec_state, self.decoders):
                tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
                next_state.append(tgt)

            if self.normalize_before:
                tgt = self.after_norm(tgt[:, -1])
            else:
                tgt = tgt[:, -1]

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                new_state = self._select_state(next_state, j)

                done[i] = (tgt[j], new_state, process[j][2])
                cache[process[j][0]] = (tgt[j], new_state)

                j += 1

        batch_states = self._create_batch_states(
            batch_states, [d[1] for d in done], [d[2] for d in done]
        )
        batch_y = torch.stack([d[0] for d in done])

        lm_tokens = to_device(
            self, torch.LongTensor([h.yseq[-1] for h in hyps]).view(final_batch)
        )

        return batch_y, batch_states, lm_tokens

    def _select_state(
        self, batch_states: List[torch.Tensor], idx: int
    ) -> List[torch.Tensor]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states: Decoder states [L x (B, max_len, D_dec)]
            idx: Index to extract state from batch of states

        Returns:
            state_idx: Decoder state for given id [L x (1, max_len, dec_dim)]

        """
        if batch_states[0] is not None:
            state_idx = [
                batch_states[layer][idx] for layer in range(len(self.decoders))
            ]
        else:
            state_idx = batch_states

        return state_idx

    def _create_batch_states(
        self,
        batch_states: List[torch.Tensor],
        l_states: List[List[torch.Tensor]],
        l_tokens: List[int],
    ) -> List[torch.Tensor]:
        """Create batch of decoder states.

        Args:
            batch_states: Decoder states [L x (B, max_len, D_dec)]
            l_states: List of single decoder states [B x [L x (1, max_len, D_dec)]]
            l_tokens: Token sequences

        Returns:
            batch_states: Decoder states [L x (B, max_len, D_dec)]

        """
        if batch_states[0] is not None:
            max_len = max([len(t) for t in l_tokens])

            for layer in range(len(self.decoders)):
                batch_states[layer] = pad_batch_state(
                    [s[layer] for s in l_states], max_len, self.blank
                )

        return batch_states


class TransformerDecoder(BaseTransformerDecoder):
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
        use_attention: bool = True,
        embed_pad: Optional[int] = None,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            use_attention=use_attention,
            embed_pad=embed_pad,
        )

        attention_dim = encoder_output_size

        if self.use_attention:
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
        else:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: TransDecoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                ),
            )


class LightweightConvolutionTransformerDecoder(BaseTransformerDecoder):
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
        use_attention: bool = True,
        embed_pad: Optional[int] = None,
    ):
        assert check_argument_types()
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
            use_attention=use_attention,
            embed_pad=embed_pad,
        )

        attention_dim = encoder_output_size

        if self.use_attention:
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
        else:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: TransDecoderLayer(
                    attention_dim,
                    LightweightConvolution(
                        wshare=conv_wshare,
                        n_feat=attention_dim,
                        dropout_rate=self_attention_dropout_rate,
                        kernel_size_str=conv_kernel_length[lnum],
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                ),
            )


class LightweightConvolution2DTransformerDecoder(BaseTransformerDecoder):
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
        use_attention: bool = True,
        embed_pad: Optional[int] = None,
    ):
        assert check_argument_types()
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
            use_attention=use_attention,
            embed_pad=embed_pad,
        )

        attention_dim = encoder_output_size

        if self.use_attention:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    LightweightConvolution2D(
                        wshare=conv_wshare,
                        n_feat=attention_dim,
                        dropout_rate=self_attention_dropout_rate,
                        kernel_size_str=conv_kernel_length[lnum],
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
        else:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: TransDecoderLayer(
                    attention_dim,
                    LightweightConvolution2D(
                        wshare=conv_wshare,
                        n_feat=attention_dim,
                        dropout_rate=self_attention_dropout_rate,
                        kernel_size_str=conv_kernel_length[lnum],
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                ),
            )


class DynamicConvolutionTransformerDecoder(BaseTransformerDecoder):
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
        use_attention: bool = True,
        embed_pad: Optional[int] = None,
    ):
        assert check_argument_types()
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
            use_attention=use_attention,
            embed_pad=embed_pad,
        )
        attention_dim = encoder_output_size

        if self.use_attention:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DecoderLayer(
                    attention_dim,
                    DynamicConvolution(
                        wshare=conv_wshare,
                        n_feat=attention_dim,
                        dropout_rate=self_attention_dropout_rate,
                        kernel_size_str=conv_kernel_length[lnum],
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
        else:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: TransDecoderLayer(
                    attention_dim,
                    DynamicConvolution(
                        wshare=conv_wshare,
                        n_feat=attention_dim,
                        dropout_rate=self_attention_dropout_rate,
                        kernel_size_str=conv_kernel_length[lnum],
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                ),
            )


class DynamicConvolution2DTransformerDecoder(BaseTransformerDecoder):
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
        use_attention: bool = True,
        embed_pad: Optional[int] = None,
    ):
        assert check_argument_types()
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
            use_attention=use_attention,
            embed_pad=embed_pad,
        )
        attention_dim = encoder_output_size

        if self.use_attention:
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
        else:
            self.decoders = repeat(
                num_blocks,
                lambda lnum: TransDecoderLayer(
                    attention_dim,
                    DynamicConvolution2D(
                        wshare=conv_wshare,
                        n_feat=attention_dim,
                        dropout_rate=self_attention_dropout_rate,
                        kernel_size=conv_kernel_length[lnum],
                        use_kernel_mask=True,
                        use_bias=conv_usebias,
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                ),
            )
