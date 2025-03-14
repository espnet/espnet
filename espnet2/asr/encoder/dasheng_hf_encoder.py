# coding=utf-8
# Copyright 2023-2024 Xiaomi Corporation and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import copy
import logging
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import torchaudio.transforms as audio_transforms
from einops.layers.torch import Rearrange
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_utils import PreTrainedModel

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug
from espnet.nets.pytorch_backend.nets_utils import roll_tensor

DASHENG_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mispeech/dasheng-base",
    "mispeech/dasheng-0.6B",
    "mispeech/dasheng-1.2B",
    # See all Dasheng models at https://huggingface.co/models?search=mispeech%2Fdasheng
]


DASHENG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mispeech/dasheng-base": "https://huggingface.co/mispeech/dasheng-base/resolve/main/config.json",
    "mispeech/dasheng-0.6B": "https://huggingface.co/mispeech/dasheng-0.6B/resolve/main/config.json",
    "mispeech/dasheng-1.2B": "https://huggingface.co/mispeech/dasheng-1.2B/resolve/main/config.json",
}


class DashengConfig(PretrainedConfig):
    model_type = "dasheng"

    def __init__(
        self,
        name: str = "dasheng-base",
        loss: str = "BCELoss",
        **kwargs,
    ):
        r"""
        Configuration class for the Dasheng model.

        Args:
            name (str, *optional*):
                Can be "dasheng-base", "dasheng-0.6B", or "dasheng-1.2B". Default to "dasheng-base".
            loss (str, *optional*):
                Name of the loss function to use. Can be any loss in `nn.modules.loss`. Default to "BCELoss".
            kwargs (dict, *optional*):
                Additional keyword arguments, see `dasheng_model.modeling_dasheng.DashengFeatureExtractor` and `dasheng_model.modeling_dasheng.AudioTransformerMAE_Encoder` for more details.
        """

        super().__init__(**kwargs)

        encoder_kwargs = dict(
            target_length=1008, patch_size=[64, 4], patch_stride=[64, 4]
        )

        if name == "dasheng-1.2B":
            encoder_kwargs["embed_dim"] = 1536
            encoder_kwargs["depth"] = 40
            encoder_kwargs["num_heads"] = 24
        elif name == "dasheng-0.6B":
            encoder_kwargs["embed_dim"] = 1280
            encoder_kwargs["depth"] = 32
            encoder_kwargs["num_heads"] = 16
        elif name == "dasheng-base":
            encoder_kwargs["embed_dim"] = 768
            encoder_kwargs["depth"] = 12
            encoder_kwargs["num_heads"] = 12
        else:
            raise ValueError(f"Unrecognized model name: {name}")
        self.name = name

        encoder_kwargs.update(
            (k, kwargs[k]) for k in set(kwargs).intersection(encoder_kwargs)
        )
        self.encoder_kwargs = {**encoder_kwargs, **kwargs}

        self.loss = loss


class DashengFeatureExtractor(SequenceFeatureExtractor):
    r"""
    DashengFeatureExtractor extracts Mel spectrogram features from audio signals.

    Args:
        f_min (int, *optional*, defaults to 0): Minimum frequency for the Mel filterbank.
        sampling_rate (int, *optional*, defaults to 16000):
            Sampling rate of the input audio signal.
        win_size (int, *optional*, defaults to 512): Window size for the STFT.
        center (bool, *optional*, defaults to `True`):
            Whether to pad the signal on both sides to center it.
        n_fft (int, *optional*, defaults to 512): Number of FFT points for the STFT.
        f_max (int, optional, *optional*): Maximum frequency for the Mel filterbank.
        hop_size (int, *optional*, defaults to 160): Hop size for the STFT.
        feature_size (int, *optional*, defaults to 64): Number of Mel bands to generate.
        padding_value (float, *optional*, defaults to 0.0): Value for padding.

    Returns:
        BatchFeature: A BatchFeature object containing the extracted features.
    """

    def __init__(
        self,
        f_min: int = 0,
        sampling_rate: int = 16000,
        win_size: int = 512,
        center: bool = True,
        n_fft: int = 512,
        f_max: Optional[int] = 8000,
        hop_size: int = 160,
        feature_size: int = 64,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.f_min = f_min
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size

        self.model_input_names = ["input_values"]

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        sampling_rate: Optional[int] = None,
        max_length: Optional[int] = 16000,
        truncation: bool = True,
        return_tensors="pt",
    ) -> BatchFeature:
        r"""
        Extracts Mel spectrogram features from an audio signal tensor.

        Args:
            x: Input audio signal tensor.
            sampling_rate (int, *optional*, defaults to `None`):
                Sampling rate of the input audio signal.
            max_length (int, *optional*, defaults to 16000):
                Maximum length of the input audio signal.
            truncation (bool, *optional*, defaults to `True`):
                Whether to truncate the input signal to max_length.
            return_tensors (str, *optional*, defaults to "pt"):
                If set to "pt", the return type will be a PyTorch tensor.

        Returns:
            BatchFeature: A dictionary containing the extracted features.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        if return_tensors != "pt":
            raise NotImplementedError(
                "Only return_tensors='pt' is currently supported."
            )

        mel_spectrogram = audio_transforms.MelSpectrogram(
            f_min=self.f_min,
            sample_rate=sampling_rate,
            win_length=self.win_size,
            center=self.center,
            n_fft=self.n_fft,
            f_max=self.f_max,
            hop_length=self.hop_size,
            n_mels=self.feature_size,
        ).to(x[0].device)
        amplitude_to_db = audio_transforms.AmplitudeToDB(top_db=120).to(x[0].device)

        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[np.newaxis, :]
            if x.ndim != 2:
                raise ValueError("np.ndarray input must be a 1D or 2D.")
            x = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.dim() != 2:
                raise ValueError("torch.Tensor input must be a 1D or 2D.")
        elif isinstance(x, (list, tuple)):
            longest_length = max(x_.shape[0] for x_ in x)
            if not truncation and max_length < longest_length:
                max_length = longest_length

            if all(isinstance(x_, np.ndarray) for x_ in x):
                if not all(x_.ndim == 1 for x_ in x):
                    raise ValueError("All np.ndarray in a list must be 1D.")

                x_trim = [x_[:max_length] for x_ in x]
                x_pad = [
                    np.pad(
                        x_,
                        (0, max_length - x_.shape[0]),
                        mode="constant",
                        constant_values=0,
                    )
                    for x_ in x_trim
                ]
                x = torch.stack([torch.from_numpy(x_) for x_ in x_pad])
            elif all(isinstance(x_, torch.Tensor) for x_ in x):
                if not all(x_.dim() == 1 for x_ in x):
                    raise ValueError("All torch.Tensor in a list must be 1D.")
                x_pad = [
                    torch.nn.functional.pad(x_, (0, max_length - x_.shape[0]), value=0)
                    for x_ in x
                ]
                x = torch.stack(x_pad)
            else:
                raise ValueError("Input list must be numpy arrays or PyTorch tensors.")
        else:
            raise ValueError(
                "Input must be a numpy array, a list of numpy arrays, a PyTorch tensor, or a list of PyTorch tensor."
            )

        x = x.float()
        x = mel_spectrogram(x)
        x = amplitude_to_db(x)
        return BatchFeature({"input_values": x})


# The functions `trunc_normal_`, `_no_grad_trunc_normal_`, `drop_path` and the module `DropPath`` are taken from timm
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def to_2tuple(x: Any) -> Tuple[Any, Any]:
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class AudioPatchEmbed(nn.Module):
    def __init__(
        self,
        input_size=224,
        patch_size=16,
        patch_stride=16,
        in_chans=1,
        embed_dim=768,
        norm_layer=None,
        flatten=False,
    ):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (
            input_size[0] // patch_stride[0],
            input_size[1] // patch_stride[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = torch.permute(
                torch.flatten(x, 2, 3), (0, 2, 1)
            )  # rearrange(x, "b c f t -> b (f t) c")
        x = self.norm(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_type="Attention",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_type = globals()[attention_type]
        self.attn = attn_type(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class AudioTransformerMAE_Encoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        patch_stride=16,
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        init_values=None,
        target_length=1012,
        pooling="mean",
        wavtransforms=None,
        spectransforms=None,
        time_patch_out: Optional[float] = None,
        freq_patch_out: Optional[float] = None,
        block_type="Block",
        attention_type="Attention",
        eval_avg="mean",
        **kwargs,
    ):
        super().__init__()
        assert pooling in ("mean", "token")
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = kwargs.get("n_mels", 64)
        init_bn = kwargs.get("init_bn", True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out
        self.pad_last = kwargs.get("pad_last", True)

        if init_bn:
            self.init_bn = nn.Sequential(
                Rearrange("b c f t -> b f c t"),
                torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
                Rearrange("b f c t -> b c f t"),
            )

        self.target_length = target_length
        self.patch_embed = AudioPatchEmbed(
            input_size=(self.n_mels, target_length),
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            flatten=False,
            patch_stride=self.patch_stride,
        )
        self.spectransforms = (
            nn.Sequential() if spectransforms is None else spectransforms
        )
        self.wavtransforms = nn.Sequential() if wavtransforms is None else wavtransforms
        self.num_patches = self.patch_embed.num_patches

        if pooling == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * 0.02
        )
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * 0.02
        )

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)
        block_function = globals()[block_type]
        self.blocks = nn.Sequential(
            *[
                block_function(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attention_type=attention_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        if hasattr(self, "cls_token") and self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_pos_embed", "cls_token", "freq_pos_embed", "token_pos_embed"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just for sin pos embed
        x = torch.permute(
            torch.flatten(x, 2, 3), (0, 2, 1)
        )  # rearrange(x, "b c f t -> b (f t) c")
        if self.pooling == "token":
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.init_bn(x) if self.init_bn is not None else x
        # Remember starting position if we pad
        padding_start = 0
        if x.shape[-1] > self.target_length:
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(
                        *x.shape[:-1], self.target_length, device=x.device
                    )
                    pad[..., : splits[-1].shape[-1]] = splits[-1]
                    padding_start = x.shape[-1] // self.patch_stride[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = torch.flatten(splits, 0, 1)  # spl b c f t-> (spl b) c f t
        else:
            n_splits = 1

        x = self.forward_features(x)
        x = torch.reshape(x, (x.shape[0] // n_splits, -1, x.shape[-1]))
        if padding_start:
            x = x[:, :padding_start, :]
        return x


class DashengPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple
    interface for downloading and loading pretrained models.
    """

    config_class = DashengConfig
    base_model_prefix = "dasheng"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class DashengModel(DashengPreTrainedModel):
    def __init__(self, config: DashengConfig, outputdim: Optional[int] = None) -> None:
        r"""
        Initializes the model.

        Args:
            config (DashengConfig): Configuration class for the model.
            outputdim (int, optional): for compatibility wiht hf, dummy.
        """
        super().__init__(config)
        self.config = config
        self.name = config.name

        self.encoder = AudioTransformerMAE_Encoder(**config.encoder_kwargs)
        self.outputdim = config.encoder_kwargs["embed_dim"]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the Dasheng model with audio features.
        The model returns hidden states.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, n_mels, sequence_length)`):
                The sequence of audio features extracted from the audio signal. Can be obtained from a raw audio waveform
                using `~transformers.DashengFeatureExtractor.__call__`.
        """
        x = torch.unsqueeze(input_values, 1)
        last_hidden_states = self.encoder(x)
        return last_hidden_states


class DashengEncoder(AbsEncoder):
    def __init__(
        self,
        input_size: int,
        model_name: str,
        roll_augment: bool = False,
        roll_interval: int = 1,
        specaug_config: Optional[Dict] = None,
        final_layer_norm: bool = False,
    ):
        """
        model_name: Name of the model to load. Must be one of the models in
            `DASHENG_PRETRAINED_MODEL_ARCHIVE_LIST`.
        roll_augment: Apply roll augmentation to the input.
        roll_interval: Interval for roll augmentation. All rolling is
            quantized to this interval.
        specaug_config: Dictionary containing parameters for SpecAugment.
            If provided, SpecAugment will be applied.
        final_layer_norm: Apply layer normalization to the output.
        """
        super().__init__()
        assert (
            model_name in DASHENG_PRETRAINED_MODEL_ARCHIVE_LIST
        ), "Invalid model name. Must be one of {}".format(
            DASHENG_PRETRAINED_MODEL_ARCHIVE_LIST
        )
        self.model = DashengModel.from_pretrained(model_name)
        self.feature_extractor = DashengFeatureExtractor.from_pretrained(model_name)
        self.pretrained_params = copy.deepcopy(self.model.state_dict())
        self.roll_augment = roll_augment
        self.roll_interval = roll_interval
        self.specaug = None
        if specaug_config is not None:
            self.specaug = SpecAug(**specaug_config)
        self.layernorm = (
            None if not final_layer_norm else nn.LayerNorm(self.model.encoder.embed_dim)
        )

    def reload_pretrained_parameters(self):
        self.model.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Dasheng model parameters reloaded!")

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if xs_pad.ndim == 2:
            xs_pad = xs_pad.unsqueeze(-1)

        assert xs_pad.ndim == 3, xs_pad.size()
        assert xs_pad.shape[-1] == 1, xs_pad.size()
        if self.roll_augment and self.training:
            xs_pad = roll_tensor(xs_pad, ilens, fixed_intervals=self.roll_interval)
        xs_pad = xs_pad.squeeze(-1)
        assert xs_pad.ndim == 2, xs_pad.size()

        x = [xs_pad[i, :ilen] for i, ilen in enumerate(ilens)]
        x = self.feature_extractor(x)  # run dasheng frontend

        if self.specaug is not None and self.training:
            x["input_values"] = self.specaug(x["input_values"])[0]

        x = self.model(**x)
        if self.layernorm is not None:
            x = self.layernorm(x)
        olens = torch.tensor(
            [x_.shape[0] for x_ in x], dtype=torch.long, device=x.device
        )
        return x, olens, None

    def output_size(self) -> int:
        return self.model.outputdim


# if __name__ == "__main__":
#     model_name = "mispeech/dasheng-base"
#     encoder = DashengEncoder(1,model_name)

# # Test forward
# xs_pad = torch.randn(2, 48000)
# ilens = torch.LongTensor([24000, 32000])
# out = encoder.forward(xs_pad, ilens)
# print(out[0].shape)
# print(out[1])
# print(out[2])
