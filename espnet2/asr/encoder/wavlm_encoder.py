# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# WavLM pretraining encoder.
#     Paper: https://arxiv.org/abs/2110.13900
#     Code: https://github.com/microsoft/unilm/tree/master/wavlm
#
# WavLM keeps HuBERT's masked cluster-prediction objective and changes two
# things: the Transformer uses a gated relative position bias on top of the
# convolutional positional embedding, and the input waveform is corrupted by
# utterance mixing (that part lives on the data path, see
# espnet2/wavlm/utterance_mixing.py).
#
# torchaudio ships the WavLM Transformer (``components._get_wavlm_encoder``) but
# only wires it into inference models (``torchaudio.models.wavlm_model``); there
# is no ``wavlm_pretrain_model`` counterpart to ``hubert_pretrain_model``. This
# module builds that missing pretraining model so the WavLM recipe can reuse the
# HuBERT mask generator, logit generator and loss unchanged.

"""WavLM pretraining encoder definition."""

import copy
import logging
from typing import List, Optional, Tuple

import torch
from torchaudio.models.wav2vec2 import components as ta_components
from torchaudio.models.wav2vec2 import model as ta_model
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class WavLMTransformer(ta_components.Transformer):
    """WavLM Transformer that masks padded keys.

    ``WavLMSelfAttention`` cannot consume the additive attention mask that
    ``torchaudio``'s Wav2Vec2/HuBERT attention takes (it asserts it is ``None``)
    and expects a ``key_padding_mask`` of shape ``(batch, src_len)`` instead. The
    surrounding ``torchaudio`` code only ever passes ``attention_mask``, so
    padded frames would either raise or be attended to. These overrides reroute
    the mask, keeping the call signatures ``HuBERTPretrainModel`` and
    ``Wav2Vec2Model`` use.

    NOTE: in this subclass ``attention_mask`` is a *key padding* mask of shape
    ``(batch, src_len)`` that is ``True`` at padded positions, as produced by
    :meth:`WavLMEncoder._preprocess`.
    """

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(
                    x,
                    attention_mask=None,
                    position_bias=position_bias,
                    key_padding_mask=attention_mask,
                )

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[torch.Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(
                    f"`num_layers` must be between [1, {len(self.layers)}]"
                )

        ret: List[torch.Tensor] = []
        position_bias = None
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(
                x,
                attention_mask=None,
                position_bias=position_bias,
                key_padding_mask=attention_mask,
            )
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


class WavLMEncoder(ta_components.Encoder):
    """Encoder whose ``_preprocess`` emits a key padding mask for WavLM.

    Same as ``torchaudio``'s ``Encoder`` except that the second return value of
    :meth:`_preprocess` is a ``(batch, src_len)`` boolean key padding mask
    (``True`` at padding) rather than an additive ``(batch, 1, src_len, src_len)``
    attention mask. See :class:`WavLMTransformer`.
    """

    def _preprocess(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[torch.Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(
                batch_size, max_len
            ) >= lengths[:, None]
            x[mask] = 0.0
        return x, mask


def wavlm_pretrain_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[List[int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    mask_prob: float,
    mask_selection: str,
    mask_other: float,
    mask_length: int,
    no_mask_overlap: bool,
    mask_min_space: int,
    mask_channel_prob: float,
    mask_channel_selection: str,
    mask_channel_other: float,
    mask_channel_length: int,
    no_mask_channel_overlap: bool,
    mask_channel_min_space: int,
    skip_masked: bool,
    skip_nomask: bool,
    num_classes: int,
    final_dim: int,
    feature_grad_mult: Optional[float],
) -> "ta_model.HuBERTPretrainModel":
    """Build a WavLM model for self-supervised pretraining.

    This is ``torchaudio.models.hubert_pretrain_model`` with the WavLM
    Transformer (gated relative position bias) in place of the HuBERT one, so
    the result is a ``torchaudio.models.HuBERTPretrainModel`` producing the same
    masked/unmasked logits and feature penalty.

    Args:
        See :py:func:`torchaudio.models.hubert_pretrain_model` for every
        argument except the two WavLM specific ones:
        encoder_num_buckets: Number of buckets for the relative position bias.
        encoder_max_distance: Maximum distance for the relative position bias.

    Returns:
        torchaudio.models.HuBERTPretrainModel
    """
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = (
            [[512, 10, 5]] + [[512, 3, 2]] * 4 + [[512, 2, 2]] * 2
        )

    feature_extractor = ta_components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias
    )
    encoder = ta_components._get_wavlm_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        num_buckets=encoder_num_buckets,
        max_distance=encoder_max_distance,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    # Swap in the padding-aware subclasses. They add no parameters or buffers,
    # only method overrides, so re-tagging the instances keeps the module tree
    # (and therefore the state_dict) identical to torchaudio's while letting the
    # builder above stay the single source of truth for the architecture.
    encoder.__class__ = WavLMEncoder
    encoder.transformer.__class__ = WavLMTransformer

    wav2vec2 = ta_model.Wav2Vec2Model(feature_extractor, encoder)
    mask_generator = ta_components.MaskGenerator(
        encoder_embed_dim,
        mask_prob,
        mask_selection,
        mask_other,
        mask_length,
        no_mask_overlap,
        mask_min_space,
        mask_channel_prob,
        mask_channel_selection,
        mask_channel_other,
        mask_channel_length,
        no_mask_channel_overlap,
        mask_channel_min_space,
    )
    logit_generator = ta_components.LogitGenerator(
        encoder_embed_dim,
        num_classes,
        final_dim,
        skip_masked,
        skip_nomask,
    )
    model = ta_model.HuBERTPretrainModel(
        wav2vec2=wav2vec2,
        mask_generator=mask_generator,
        logit_generator=logit_generator,
        feature_grad_mult=feature_grad_mult,
    )
    # Initialize the model for pre-training. This walks the Transformer with
    # `_init_transformer_params`, covering WavLMSelfAttention's Linear and
    # Embedding (relative position bias) sub-modules; its fused
    # MultiheadAttention `in_proj_weight` keeps torch's own xavier_uniform_
    # init, as in `torchaudio.models.wavlm_model`.
    model.apply(ta_model._init_hubert_pretrain_model)
    return model


class TorchAudioWavLMPretrainEncoder(AbsEncoder):
    """Torch Audio WavLM pretraining encoder module.

    Same interface and defaults as
    :class:`espnet2.asr.encoder.hubert_encoder.TorchAudioHuBERTPretrainEncoder`,
    with the WavLM Transformer (gated relative position bias) instead of the
    HuBERT one. The two extra arguments are:

    Args:
        encoder_num_buckets: Number of buckets for the relative position bias.
        encoder_max_distance: Maximum distance for the relative position bias.

    For every other argument please refer to
    :class:`espnet2.asr.encoder.hubert_encoder.TorchAudioHuBERTPretrainEncoder`
    and https://pytorch.org/audio/stable/generated/torchaudio.models.wavlm_model.html
    """

    @typechecked
    def __init__(
        self,
        input_size: Optional[int] = None,
        extractor_mode: str = "group_norm",
        extractor_conv_layer_config: Optional[List[List[int]]] = [
            [512, 10, 5],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 2, 2],
            [512, 2, 2],
        ],
        extractor_conv_bias: bool = False,
        encoder_embed_dim: int = 768,
        encoder_projection_dropout: float = 0.1,
        encoder_pos_conv_kernel: int = 128,
        encoder_pos_conv_groups: int = 16,
        encoder_num_layers: int = 12,
        encoder_num_heads: int = 12,
        encoder_num_buckets: int = 320,
        encoder_max_distance: int = 800,
        encoder_attention_dropout: float = 0.1,
        encoder_ff_interm_features: int = 3072,
        encoder_ff_interm_dropout: float = 0.0,
        encoder_dropout: float = 0.1,
        encoder_layer_norm_first: bool = False,
        encoder_layer_drop: float = 0.05,
        mask_prob: float = 0.8,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        mask_length: int = 10,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_prob: float = 0.0,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        mask_channel_length: int = 10,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        skip_masked: bool = False,
        skip_nomask: bool = False,
        num_classes: int = 100,
        final_dim: int = 256,
        feature_grad_mult: Optional[float] = 0.1,
        finetuning: bool = False,
        freeze_encoder_updates: int = 0,
    ):
        super().__init__()

        self._output_size = encoder_embed_dim

        self.wavlm_pretrain_model = wavlm_pretrain_model(
            extractor_mode=extractor_mode,
            extractor_conv_layer_config=extractor_conv_layer_config,
            extractor_conv_bias=extractor_conv_bias,
            encoder_embed_dim=encoder_embed_dim,
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_pos_conv_kernel=encoder_pos_conv_kernel,
            encoder_pos_conv_groups=encoder_pos_conv_groups,
            encoder_num_layers=encoder_num_layers,
            encoder_num_heads=encoder_num_heads,
            encoder_num_buckets=encoder_num_buckets,
            encoder_max_distance=encoder_max_distance,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_features=encoder_ff_interm_features,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_first=encoder_layer_norm_first,
            encoder_layer_drop=encoder_layer_drop,
            mask_prob=mask_prob,
            mask_selection=mask_selection,
            mask_other=mask_other,
            mask_length=mask_length,
            no_mask_overlap=no_mask_overlap,
            mask_min_space=mask_min_space,
            mask_channel_prob=mask_channel_prob,
            mask_channel_selection=mask_channel_selection,
            mask_channel_other=mask_channel_other,
            mask_channel_length=mask_channel_length,
            no_mask_channel_overlap=no_mask_channel_overlap,
            mask_channel_min_space=mask_channel_min_space,
            skip_masked=skip_masked,
            skip_nomask=skip_nomask,
            num_classes=num_classes,
            final_dim=final_dim,
            feature_grad_mult=feature_grad_mult,
        )
        self.pretrained_params = copy.deepcopy(self.wavlm_pretrain_model.state_dict())

        self.finetuning = finetuning
        if finetuning:
            for p in self.wavlm_pretrain_model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
        self.register_buffer("global_step", torch.tensor([0], dtype=torch.long))
        self.freeze_encoder_updates = freeze_encoder_updates

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_length: torch.Tensor = None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward WavLM Pretrain Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            ys_pad: k-means cluster targets (B, L')
            prev_states: Not to be used now.
        Returns:
            (logit_m, logit_u, feature_penalty) while pretraining, otherwise the
            encoded sequence and its lengths.
        """
        if not self.finetuning:
            return self._pretraining_forward(xs_pad, ilens, ys_pad)
        else:
            if self.training:
                return self._finetuning_forward(xs_pad, ilens)
            else:
                return self._eval_forward(xs_pad, ilens)

    def _pretraining_forward(self, xs_pad, ilens, ys_pad):
        assert ys_pad is not None
        (
            logit_m,
            logit_u,
            feature_penalty,
        ) = self.wavlm_pretrain_model.forward(xs_pad, ys_pad, ilens)

        return logit_m, logit_u, feature_penalty

    def _finetuning_forward(self, xs_pad, ilens):
        def get_padding_mask(input, lengths):
            """get_padding_mask() from torchaudio.models.wav2vec2.components"""
            batch_size, max_len, _ = input.shape
            mask = (
                torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
                >= lengths[:, None]
            )
            return mask

        # manually add the steps. It is not accurate.
        self.global_step += 1
        if self.global_step <= self.freeze_encoder_updates:
            with torch.no_grad():
                x, out_len = self.wavlm_pretrain_model.wav2vec2.feature_extractor(
                    xs_pad, ilens
                )
                padding_mask = get_padding_mask(x, out_len)
                (
                    x,
                    attention_mask,
                ) = self.wavlm_pretrain_model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.wavlm_pretrain_model.mask_generator(x, padding_mask)
                x = self.wavlm_pretrain_model.wav2vec2.encoder.transformer(
                    x, attention_mask=attention_mask
                )
        else:
            with torch.no_grad():
                x, out_len = self.wavlm_pretrain_model.wav2vec2.feature_extractor(
                    xs_pad, ilens
                )
                padding_mask = get_padding_mask(x, out_len)

            (
                x,
                attention_mask,
            ) = self.wavlm_pretrain_model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.wavlm_pretrain_model.mask_generator(x, padding_mask)
            x = self.wavlm_pretrain_model.wav2vec2.encoder.transformer(
                x, attention_mask=attention_mask
            )
        return x, (~padding_mask).long().sum(dim=1), None

    def _eval_forward(self, xs_pad, ilens):
        x, lengths = self.wavlm_pretrain_model.wav2vec2.feature_extractor(xs_pad, ilens)
        x = self.wavlm_pretrain_model.wav2vec2.encoder(x, lengths)
        return x, lengths, None

    def reload_pretrained_parameters(self):
        self.wavlm_pretrain_model.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained WavLM model parameters reloaded!")
