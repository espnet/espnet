# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Adapted from Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq

# This code is adapted from the original BEATs implementation and
#  can be used to pre-train/and or fine-tune BEATs model.
# --------------------------------------------------------

import logging
import math
import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
from packaging.version import parse as V
from torch.nn import LayerNorm, Parameter

try:
    from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerConfig,
        Wav2Vec2ConformerEncoder,
    )

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, roll_tensor

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class BeatsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # patch size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = (
            1.0  # ratio for layer-wise gradient decay
        )
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # relative position embedding
        self.relative_position_embedding: bool = (
            False  # apply relative position embedding
        )
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = (
            1280  # maximum distance for relative position embedding
        )
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BeatsEncoder(AbsEncoder):
    """BEATs: Audio Pre-Training with Acoustic Tokenizers.

    (https://arxiv.org/abs/2212.09058)
    Args:
        beats_ckpt_path: Path to a pretrained Beats checkpoint. If
            `beats_config` is provided and it does not match the
            config in the checkpoint, code might throw an error.
        max_layer: Propagate input through all layers for encoding
            if None. Otherwise use upto `max_layer`.
        downsampling_rate: Downsampling rate for the encoder. Applied if > 1.
        adapter_config: Path to a config file for the wav2vec2 adapter.
        use_weighted_representation: Use weighted representations
            from max_layer if True. Weights are randomly initialized.
        beats_config: `BeatsConfig` object. If provided, we will try
            to override the config in the checkpoint. This can be used
            to change dropouts etc for fine-tuning the model while
            starting from a pretrained checkpoint.
        specaug_config: Dictionary containing parameters for SpecAugment.
            If provided, SpecAugment will be applied.
        add_positional_information: Add learned positional embeddings.
        max_positions: Maximum number of positions for positional embeddings.
            Required if `add_positional_information` is True.
        roll_augment: Apply roll augmentation to the input.
        roll_interval: Interval for roll augmentation. All rolling is
            quantized to this interval.
    """

    def __init__(
        self,
        input_size: int,
        beats_ckpt_path: str = None,
        max_layer: int = None,
        downsampling_rate: int = 1,
        adapter_config: str = "",
        use_weighted_representation: bool = False,
        beats_config: Optional[Dict] = None,
        specaug_config: Optional[Dict] = None,
        add_positional_information: bool = False,
        max_positions: Optional[int] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        roll_augment: bool = False,
        roll_interval: int = 1600,
    ) -> None:
        super().__init__()

        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
        self.max_layer = max_layer
        self.beats_ckpt_path = beats_ckpt_path
        self.roll_augment = roll_augment
        self.roll_interval = roll_interval

        # Four cases for loading Beats config:
        # 1. No checkpoint and no config: Default config
        # 2. Checkpoint and no user-provided config: Load config from
        #    checkpoint
        # 3. Checkpoint and user-provided config: Merge the two, but
        #    override with user-provided config
        # 4. No checkpoint and user-provided config: Use user-provided config
        if adapter_config or add_positional_information:
            # We need transformers library for adapter and positional embeddings
            if not is_transformers_available:
                raise ImportError(
                    "`transformers` is not available. Please install it "
                    " via `pip install transformers` or"
                    " `cd /path/to/espnet/tools && "
                    ". ./activate_python.sh"
                    " && ./installers/install_transformers.sh`."
                )
        config = BeatsConfig()  # Default config
        if beats_ckpt_path and beats_config:
            logging.warning(
                "Both pretrained checkpoint and config are provided."
                " We will override ckpt config with user-provided config."
            )
        self.loaded_state_dict_ = None
        if beats_ckpt_path is not None:
            self.loaded_state_dict_ = torch.load(beats_ckpt_path)
            logging.info(f"Loaded Beats pretrained config from {beats_ckpt_path}.")
            config = BeatsConfig(self.loaded_state_dict_["cfg"])
        if beats_config is not None:
            config.update(beats_config)
            logging.info("Overriding Beats config with user-provided config.")

        self.specaug = None
        if specaug_config is not None:
            self.specaug = SpecAug(**specaug_config)

        self._output_size = config.encoder_embed_dim

        self.embed = config.embed_dim
        self.input_patch_size = config.input_patch_size
        self.post_extract_proj = (
            nn.Linear(self.embed, config.encoder_embed_dim)
            if self.embed != config.encoder_embed_dim
            else None
        )
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=config.conv_bias,
        )
        self.dropout_input = nn.Dropout(config.dropout_input)
        assert not config.deep_norm or not config.layer_norm_first

        self.encoder = TransformerEncoder(config)
        self.layer_norm = LayerNorm(self.embed)

        self.use_weighted_representation = use_weighted_representation
        if self.use_weighted_representation:
            if self.max_layer is None:
                logging.warning(
                    f"max_layer must be provided when using weighted"
                    f" representations. Set to {config.encoder_layers-1}."
                )
                self.max_layer = config.encoder_layers - 1  # 0 based index
            self.layer_weights = nn.Parameter(
                torch.ones((self.max_layer + 1, 1)), requires_grad=True
            )

        # Downsampling modules
        self.encoder_downsample_rate = downsampling_rate
        self.downsample_conv = None
        if self.encoder_downsample_rate > 1:
            self.downsample_conv = nn.Conv1d(
                in_channels=config.encoder_embed_dim,
                out_channels=config.encoder_embed_dim,
                kernel_size=int(
                    round(self.encoder_downsample_rate * 1.5)
                ),  # kernel multiplier from Shih-Lun's code
                stride=self.encoder_downsample_rate,
            )

        # Adapter module
        self.conformer_adapter = None
        if adapter_config:
            conformer_config = Wav2Vec2ConformerConfig.from_json_file(adapter_config)
            self.conformer_adapter = Wav2Vec2ConformerEncoder(conformer_config)

        # Positional embeddings applied before cross-attention with decoder.
        self.cross_embed_positions = None
        if add_positional_information:
            assert (
                max_positions is not None
            ), "max_positions must be provided in the config."
            learned_pos_dim = (
                config.encoder_embed_dim
                if not self.conformer_adapter
                else self.conformer_adapter.config.hidden_size
            )
            self.cross_embed_positions = BartLearnedPositionalEmbedding(
                max_positions, learned_pos_dim
            )
        # FIXME(shikhar): This is a hack to make the model compatible with
        # small audio inputs, without this the window sizes become larger
        # than audio. We should add an option to use this via the config.
        self.min_input_length_at_16khz = 3200

    def reload_pretrained_parameters(self):
        """Initialization function for Beats.

        This must be called last in the initialization procedure.
        The initialization occurs in three steps:
        1. ESPnet initializes all modules.
        2. This function initializes Beats encoder overriding 1.
        3. Optionally, if we have the pretrained checkpoint, we load the
            weights from the checkpoint overriding 2 and 1.
        """
        logging.info("Beats Initialization function called.")
        if self.post_extract_proj:
            torch.nn.init.xavier_normal_(self.post_extract_proj.weight)
            if self.post_extract_proj.bias is not None:
                torch.nn.init.constant_(self.post_extract_proj.bias, 0)
        torch.nn.init.xavier_normal_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            torch.nn.init.constant_(self.patch_embedding.bias, 0)

        # Beats has different initialization from ESPnet for other modules,
        #  so override.
        self.encoder.apply(init_bert_params)
        if self.loaded_state_dict_ is not None:

            load_info = self.load_state_dict(
                self.loaded_state_dict_["model"], strict=False
            )
            # strict=False to ignore Weights in the predictor
            logging.info(
                f"Loaded Beats pretrained model. Following keys were missing"
                f" in your custom model: {load_info.missing_keys}. "
                f"Follwing keys could not be loaded from the pretrained"
                f"checkpoint: {load_info.unexpected_keys}."
                "It is expected to have 'predictor' listed above if you are "
                "fine-tuning with only the Beats backbone."
            )

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward padding mask. Featuires: BTC, padding_mask: BT."""
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)  # remove totally empty sequences
        return padding_mask

    def preprocess(
        self,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess raw audio."""
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2**15  # float32 to int16
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        return fbank

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Wrapper for compatibility with ESPnets' AbsEncoder Interface.

        Args:
            xs_pad: (B, T)
            ilens: (B,)
            prev_states: None
        Returns:
            audio_representation: (B, T, D)
            output_lens: (B,)
            masks: None
        """
        if self.roll_augment and self.training:
            xs_pad = roll_tensor(
                xs_pad.unsqueeze(-1), ilens, fixed_intervals=self.roll_interval
            ).squeeze(-1)
        # NOTE(shikhar): If xs is not provided then the operation is costly,
        # because this function tries to create a tensor of size maxlen x maxlen.
        # Therfore, we unsqueeze and then squeeze tensors.
        mask = make_pad_mask(
            lengths=ilens, xs=xs_pad.unsqueeze(-1).unsqueeze(-1), length_dim=1
        ).to(xs_pad.device)
        # Adjust shapes to be compatible with Beats code
        mask = mask.squeeze(-1).squeeze(-1)
        audio_representation, mask = self.extract_features(
            xs_pad,
            mask,
            max_layer=self.max_layer,
        )
        output_lens = (~mask).sum(-1)
        return audio_representation, output_lens, None

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        max_layer: Optional[int] = None,
    ):
        """Extract features from raw audio."""

        if source.size(1) < self.min_input_length_at_16khz:
            logging.warning(
                f"Input shape: {source.shape}. This is less than"
                f" the minimum size of {self.min_input_length_at_16khz}."
            )
            # repeat the input to make it at least min_length
            source = torch.cat(
                [source] * (self.min_input_length_at_16khz // source.size(1) + 1), dim=1
            )

        with autocast(False):
            fbank = self.preprocess(source)

            if self.specaug is not None and self.training:
                fbank = self.specaug(fbank)[0]

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1).float()
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            # features is BTC
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        features, layer_results = self.encoder(
            features, padding_mask=padding_mask, layer=max_layer
        )

        if max_layer is not None:
            features = layer_results[max_layer][0].transpose(
                0, 1
            )  # use the output from the max_layer

        if self.use_weighted_representation:
            repr_layer_weights = nn.functional.softmax(self.layer_weights, dim=-2)
            assert (
                max_layer is not None
            ), "max_layer must not be None when using weighted representations."
            features = (
                torch.stack(
                    [
                        layer_result_i.transpose(0, 1)
                        for layer_result_i, _ in layer_results[: max_layer + 1]
                    ],
                    dim=-2,
                )
                * repr_layer_weights
            )
            features = features.sum(dim=-2)  # BTC

        if self.downsample_conv is not None:
            features = self.downsample_conv(features.transpose(1, 2)).transpose(
                1, 2
            )  # BTC
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.conformer_adapter:
            # to handle incompatibility btw torch & huggingface
            conformer_attn_mask = ~padding_mask
            # run through conformer
            features = self.conformer_adapter(
                features,
                attention_mask=conformer_attn_mask,
            ).last_hidden_state

        if self.cross_embed_positions is not None:
            features = features + self.cross_embed_positions(features)

        return features, padding_mask


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, config):
        super().__init__()

        self.dropout = config.dropout
        self.embedding_dim = config.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=config.conv_pos,
            padding=config.conv_pos // 2,
            groups=config.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (config.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(
            self.pos_conv, SamePad(config.conv_pos), nn.GELU()
        )

        if hasattr(config, "relative_position_embedding"):
            self.relative_position_embedding = config.relative_position_embedding
            self.num_buckets = config.num_buckets
            self.max_distance = config.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=config.encoder_ffn_embed_dim,
                    num_attention_heads=config.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=config.attention_dropout,
                    activation_dropout=config.activation_dropout,
                    activation_fn=config.activation_fn,
                    layer_norm_first=config.layer_norm_first,
                    deep_norm=config.deep_norm,
                    has_relative_attention_bias=self.relative_position_embedding,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=config.gru_rel_pos,
                    encoder_layers=config.encoder_layers,
                )
                for i in range(config.encoder_layers)
            ]
        )
        if self.relative_position_embedding:
            for i in range(1, config.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[
                    0
                ].self_attn.relative_attention_bias

        self.layer_norm_first = config.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = config.encoder_layerdrop

        self.apply(init_bert_params)

        if config.deep_norm:
            deep_norm_beta = math.pow(8 * config.encoder_layers, -1 / 4)
            for i in range(config.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)

        self.layer_wise_gradient_decay_ratio = getattr(
            config, "layer_wise_gradient_decay_ratio", 1
        )

    def forward(self, x, padding_mask=None, layer=None):
        """Forward pass."""
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        """Extract features from the input sequence."""

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply((x, self.layer_wise_gradient_decay_ratio))
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    pos_bias=pos_bias,
                )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        deep_norm: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        encoder_layers: int = 0,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        """Forward pass."""
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        has_relative_attention_bias=False,
        num_buckets=32,
        max_distance=128,
        gru_rel_pos=False,
        rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Initiate parameters in the transformer model."""
        logging.info("Initiate parameters in the MultiheadAttention module.")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(
                relative_positions, torch.zeros_like(relative_positions)
            )

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_positions, relative_postion_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute relative position bias."""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = (
                position_bias.unsqueeze(0)
                .repeat(bsz, 1, 1, 1)
                .view(bsz * self.num_heads, tgt_len, src_len)
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.k_head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (
            attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]
        ) * alpha
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = (
                    q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                    * alpha
                    / self.scaling
                )
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer)
                    .view(_B, _H, _L, 2, 4)
                    .sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = (
                    gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias
                )

            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())

            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[torch.Tensor],
        prev_key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[torch.Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        """No op"""
        return attn_weights


def init_bert_params(module):
    """Initialize the weights specific to the BERT Model.

    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        logging.info("Intializing Linear Layer")
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        logging.info("Intializing Embedding Layer")
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        logging.info("Intializing Multihead Attention")
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GradMultiply(torch.autograd.Function):
    """A gradient modification function that scales the gradient by a fixed scalar"""

    @staticmethod
    def forward(ctx, i):
        """Forward pass"""
        x, scale = i
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        """Backward pass"""
        return grad * ctx.scale


class SamePad(nn.Module):
    """Change input tensor shape according to the kernel size and type of LM"""

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """Forward pass"""
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    """Swish activation function"""

    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """Forward pass"""
        return x * self.act(x)


class GLU_Linear(nn.Module):
    """GLU Linear layer"""

    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        """Forward pass"""
        # to be consistent with GLU_Linear, we assume the input always has the
        # #channel (#dim) in the last dimension of the tensor, so need to
        # switch the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (
                x[:, :, 0 : self.output_dim]
                * x[:, :, self.output_dim : self.output_dim * 2]
            )
        else:
            x = x[:, :, 0 : self.output_dim] * self.glu_act(
                x[:, :, self.output_dim : self.output_dim * 2]
            )

        return x


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn("--activation-fn=gelu_fast has been renamed to gelu_accurate")
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def quant_noise(module, p, block_size):
    """Wraps modules and applies quantization noise to the weights for

    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
