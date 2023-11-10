# Copyright 2023
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# The original AVHubert work is in:
#     Paper: https://arxiv.org/pdf/2201.02184.pdf
#     Original code: https://github.com/facebookresearch/av_hubert


"""Encoder definition."""
import contextlib
import copy
import logging
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(
            kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
        ),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def time_masking(xs_pad, min_T=5, max_T=20):
    """Masking Contiguous Frames with random length of [min_T, max_T]"""
    batch_size = xs_pad.size(0)
    mask = torch.ones_like(xs_pad)
    for b in range(batch_size):
        width = min(random.randint(min_T, max_T), xs_pad.size(1))
        start = random.randint(0, xs_pad.size(1) - width)
        mask[b, start : start + width] = 0.0
    return xs_pad * mask.to(xs_pad.device)


# avhubert_url(noise_large):
# 'https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt'
# avhubert_url(noise_base):
# 'https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/base_vox_iter5.pt'
class FairseqAVHubertEncoder(AbsEncoder):
    """FairSeq AVHubert pretrained encoder module

    Args:
        input_size: input dim
        avhubert_url: download link for pre-trained avhubert model
        avhubert_dir_path: dir_path for downloading pre-trained avhubert model
    """

    def __init__(
        self,
        input_size: int = 1,
        avhubert_url: str = "./",
        avhubert_dir_path: str = "./",
        freeze_finetune_updates: int = 0,
        encoder_embed_dim: int = 1024,
        encoder_layerdrop: float = 0.05,
        dropout_input: float = 0.1,
        dropout_features: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        feature_grad_mult: float = 0.1,
        activation_dropout: float = 0.0,
        wav_input: bool = False,
        layer_norm_first: bool = True,
        audio_feat_dim: int = 104,
        encoder_layers: int = 24,
        encoder_ffn_embed_dim: int = 4096,
        encoder_attention_heads: int = 16,
        extracted: bool = False,
        pretrain: bool = True,
    ):
        assert check_argument_types()
        super().__init__()

        self._output_size = encoder_embed_dim
        self.extracted = extracted

        arg_overrides = {
            "encoder_embed_dim": encoder_embed_dim,
            "encoder_layerdrop": encoder_layerdrop,
            "dropout_input": dropout_input,
            "dropout_features": dropout_features,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "feature_grad_mult": feature_grad_mult,
            "activation_dropout": activation_dropout,
            "wav_input": wav_input,
            "layer_norm_first": layer_norm_first,
            "audio_feat_dim": audio_feat_dim,
            "encoder_layers": encoder_layers,
            "encoder_ffn_embed_dim": encoder_ffn_embed_dim,
            "encoder_attention_heads": encoder_attention_heads,
        }
        default_cfg = AVHubertConfig()
        for arg_name, arg_val in arg_overrides.items():
            setattr(default_cfg, arg_name, arg_val)

        model = AVHubertModel.build_model(cfg=default_cfg)

        if pretrain:
            self.avhubert_model_path = download_avhubert(
                avhubert_url,
                avhubert_dir_path,
            )

            ckpt = torch.load(
                self.avhubert_model_path,
                map_location=torch.device("cpu"),
            )
            state = {
                k: v
                for k, v in ckpt["model"].items()
                if "label_embs_concat" not in k and "final_proj" not in k
            }
            del ckpt
            model.load_state_dict(state)
        else:
            logging.info(
                "Training from scratch without \
                         using pre-trained AV-HuBERT model"
            )

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.encoders = model

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: Dict[str, torch.Tensor],
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward AVHubert Encoder.
        Args:
            xs_pad[video]: input tensor (B, 1, L, H, W)
            xs_pad[audio]: input tensor (B, D, L)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        if not self.extracted:
            if "video" in xs_pad:
                masks = make_pad_mask(ilens, length_dim=2).to(xs_pad["video"].device)
            elif "audio" in xs_pad:
                masks = make_pad_mask(ilens, length_dim=2).to(xs_pad["audio"].device)
            else:
                ValueError(f"Input should have video or audio")

            ft = self.freeze_finetune_updates <= self.num_updates

            if self.num_updates <= self.freeze_finetune_updates:
                self.num_updates += 1
            elif ft and self.num_updates == self.freeze_finetune_updates + 1:
                self.num_updates += 1
                logging.info("Start fine-tuning AVhubert parameters!")
            else:
                self.num_updates += 1
            with torch.no_grad() if not ft else contextlib.nullcontext():
                enc_outputs = self.encoders.extract_finetune(
                    xs_pad,
                    padding_mask=masks,
                )
        else:
            masks = make_pad_mask(ilens, length_dim=1).to(xs_pad.device)
            ft = self.freeze_finetune_updates <= self.num_updates

            if self.training:
                xs_pad = time_masking(xs_pad)

            if self.num_updates <= self.freeze_finetune_updates:
                self.num_updates += 1
            elif ft and self.num_updates == self.freeze_finetune_updates + 1:
                self.num_updates += 1
                logging.info("Start fine-tuning AVhubert parameters!")
            else:
                self.num_updates += 1
            with torch.no_grad() if not ft else contextlib.nullcontext():
                enc_outputs = self.encoders.forward_transformer(
                    xs_pad,
                    padding_mask=masks,
                )

        xs_pad = enc_outputs[0]
        masks = enc_outputs[1]

        # save gpu memory
        del enc_outputs

        olens = (~masks).sum(dim=1)

        return xs_pad, olens, None

    def forward_fusion(
        self,
        xs_pad: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        audio_feats = self.encoders.forward_audio(xs_pad["audio"])
        video_feats = self.encoders.forward_video(xs_pad["video"])
        return self.encoders.modality_fusion(audio_feats, video_feats)

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained AVHubert model parameters reloaded!")


@dataclass
class AVHubertConfig:
    """Configuration from original AVHubert Github"""

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: str = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: str = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: str = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(
        default="prelu", metadata={"help": "relu type for resnet"}
    )
    resnet_weights: Optional[str] = field(
        default=None, metadata={"help": "resnet weights"}
    )
    sim_type: str = field(default="cosine", metadata={"help": "similarity type"})

    sub_encoder_layers: int = field(
        default=0, metadata={"help": "number of transformer layers for single modality"}
    )
    audio_feat_dim: int = field(
        default=-1, metadata={"help": "audio feature dimension"}
    )
    modality_dropout: float = field(default=0, metadata={"help": "drop one modality"})
    audio_dropout: float = field(default=0, metadata={"help": "drop audio feature"})
    modality_fuse: str = field(
        default="concat", metadata={"help": "fusing two modalities: add,concat"}
    )
    selection_type: str = field(
        default="same_other_seq",
        metadata={
            "help": "type of selectig images,"
            "same_other_seq: replace masked span with span from another sequence,"
            "same_seq: repace masked span with span of the same sequence"
        },
    )
    masking_type: str = field(
        default="input", metadata={"help": "input or feature masking"}
    )

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings " "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights " "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={"help": "scale embedding"})


class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x


class AVHubertModel(nn.Module):
    def __init__(self, cfg: AVHubertConfig, **kwargs) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        try:
            from fairseq.modules import LayerNorm
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(
            resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg
        )
        self.feature_extractor_video = SubModel(
            resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg
        )
        self.modality_dropout, self.audio_dropout = (
            cfg.modality_dropout,
            cfg.audio_dropout,
        )
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == "concat":
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == "add":
            self.embed = cfg.encoder_embed_dim
        else:
            ValueError(f"unknown fusion method: {self.modality_fuse}")
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = (
            cfg.mask_prob_image,
            cfg.mask_prob_audio,
        )
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = (
            cfg.mask_length_image,
            cfg.mask_length_audio,
        )
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_()
            if self.masking_type == "input"
            else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    @classmethod
    def build_model(cls, cfg: AVHubertConfig):
        """Build a new model instance."""

        kwargs = {}
        model = cls(cfg, **kwargs)
        return model

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_finetune(
        self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None
    ):
        """Forward AVHubert Pretrain Encoder.
        Args:
            source['video']: input tensor (B, 1, L, H, W)
            source['audio']: input tensor (B, F, L)
            padding_mask: input tensor (B, L)
        Returns:
            encoded tensor and mask
        """
        src_audio, src_video = source["audio"], source["video"]

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(
                src_audio, modality="audio"
            )  # features: [B, F, T]
            features_video = features_audio.new_zeros(
                features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1)
            )
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality="video")
            features_audio = features_video.new_zeros(
                features_video.size(0), self.encoder_embed_dim, features_video.size(-1)
            )
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality="video")
            features_audio = self.forward_features(
                src_audio, modality="audio"
            )  # features: [B, F, T]
        else:
            ValueError("Both audio and video is None")

        if self.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == "add":
            features = features_audio + features_video
        else:
            ValueError(f"unknown fusion method: {self.modality_fuse}")

        features = features.transpose(1, 2)  # B, 2F, T -> B, T, 2F
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return x, padding_mask

    def forward_audio(self, source_audio):
        with torch.no_grad():
            features_audio = self.forward_features(
                source_audio, modality="audio"
            )  # features: [B, F, T]
        return features_audio

    def forward_video(self, source_video):
        with torch.no_grad():
            features_video = self.forward_features(
                source_video, modality="video"
            )  # features: [B, F, T]
        return features_video

    def modality_fusion(self, features_audio, features_video):
        if features_audio is None and features_video is not None:
            features_video = features_audio.new_zeros(
                features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1)
            )
        elif features_video is None and features_audio is not None:
            features_audio = features_video.new_zeros(
                features_video.size(0), self.encoder_embed_dim, features_video.size(-1)
            )
        else:
            features_video = features_video
            features_audio = features_audio

        if self.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == "add":
            features = features_audio + features_video
        else:
            ValueError(f"unknown fusion method: {self.modality_fuse}")

        return features

    def forward_transformer(self, source, padding_mask=None, output_layer=None):
        """Forward AVHubert Pretrain Encoder (without frontend).
        Assume the source is already fused feature.
        Args:
            source: input tensor (B, L, D*2)
            padding_mask: input tensor (B, L)
        Returns:
            encoded tensor and mask
        """
        features = source
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return x, padding_mask


def download_avhubert(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    if not os.path.exists(model_path):
        with FileLock(model_path + ".lock"):
            torch.hub.download_url_to_file(model_url, model_path)
            logging.info(f"AVHubert model downloaded {model_path}")
    else:
        logging.info(f"AVHubert model {model_path} already exists.")

    return model_path


class TransformerEncoder(nn.Module):
    """From AVHubert github"""

    def __init__(self, args):
        super().__init__()
        try:
            from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
            from fairseq.modules import LayerNorm
            from fairseq.modules.transformer_sentence_encoder import init_bert_params
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
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

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu"):
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception("relu type not implemented")

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        relu_type="relu",
        gamma_zero=False,
        avg_pool_downsample=False,
    ):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = (
            downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        )

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResEncoder(nn.Module):
    def __init__(self, relu_type, weights):
        super(ResEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == "prelu"
            else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        if weights is not None:
            logger.info(f"Load {weights} for resnet")
            std = torch.load(weights, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
            frontend_std, trunk_std = OrderedDict(), OrderedDict()
            for key, val in std.items():
                new_key = ".".join(key.split(".")[1:])
                if "frontend3D" in key:
                    frontend_std[new_key] = val
                if "trunk" in key:
                    trunk_std[new_key] = val
            self.frontend3D.load_state_dict(frontend_std)
            self.trunk.load_state_dict(trunk_std)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self.threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        return x

    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch * s_time, n_channels, sx, sy)


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
