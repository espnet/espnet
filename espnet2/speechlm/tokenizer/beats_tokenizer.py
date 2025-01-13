# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# Based on VQGAN code bases
# https://github.com/CompVis/taming-transformers
# This code is adapted from the original BEATs implementation and
#  can be used to tokenize audio using a BEATs model.
# --------------------------------------------------------

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
from torch.nn import LayerNorm

from espnet2.asr.encoder.beats_encoder import BeatsConfig, BeatsEncoder


class BeatsTokenizerConfig(BeatsConfig):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # quantizer
        self.quant_n: int = 1024  # codebook number in quantizer
        self.quant_dim: int = 256  # codebook dimension in quantizer
        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BeatsTokenizer(BeatsEncoder):
    def __init__(
        self,
        beats_tokenizer_ckpt_path: str = None,
        tokenizer_config: Optional[Dict] = None,
        max_layer: int = None,
        use_weighted_representation: bool = False,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> None:
        if beats_tokenizer_ckpt_path is None:
            logging.warning(
                "No checkpoint path provided!!"
                "We will encode audio using a randomly initialized model."
            )
        super().__init__(
            input_size=1,
            beats_ckpt_path=beats_tokenizer_ckpt_path,
            max_layer=max_layer,
            use_weighted_representation=use_weighted_representation,
            beats_config=tokenizer_config,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
        )

        config = BeatsTokenizerConfig()  # default config
        if self.loaded_state_dict_:
            config = BeatsTokenizerConfig(self.loaded_state_dict_["cfg"])
        if tokenizer_config:
            config.update(tokenizer_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=config.quant_n, embedding_dim=config.quant_dim, beta=1.0, decay=0.99
        )
        self.quantize_layer = nn.Sequential(
            nn.Linear(
                self._output_size,
                self._output_size,
            ),
            nn.Tanh(),
            nn.Linear(self._output_size, config.quant_dim),
        )
        self.reload_pretrained_parameters()

    def reload_pretrained_parameters(self):
        super().reload_pretrained_parameters()
        logging.info("Beats Tokenizer initialization function called.")
        if self.loaded_state_dict_ is not None:
            load_info = self.load_state_dict(
                self.loaded_state_dict_["model"], strict=False
            )
            logging.info(
                f"Loaded Beats tokenizer pretrained model."
                f"Following keys were missing"
                f" in your custom model: {load_info.missing_keys}. "
                f"Follwing keys could not be loaded from the pretrained"
                f"checkpoint: {load_info.unexpected_keys}."
            )

    def encode(
        self,
        xs_pad: torch.Tensor,
        ilens: Optional[torch.Tensor] = None,
    ):
        if ilens is None:
            assert (
                xs_pad.size(0) == 1
            ), "Batch size must be 1. Otherwise, please provide ilens."
            ilens = torch.tensor([xs_pad.size(1)], dtype=torch.long).to(xs_pad.device)
        x, _, _ = self(xs_pad, ilens)
        quantize_input = self.quantize_layer(x)
        quantize_feature, embed_ind = self.quantize(quantize_input)
        return embed_ind


class NormEMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embed,
        embedding_dim,
        beta,
        decay=0.99,
        kmeans_init=False,
        eps=1e-5,
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z):
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encoding_indices = encoding_indices.view(z.shape[:-1])
        return z_q, encoding_indices


class EmbeddingEMA(nn.Module):
    def __init__(
        self,
        num_tokens,
        codebook_dim,
        decay=0.99,
        eps=1e-5,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        weight = l2norm(weight)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)
