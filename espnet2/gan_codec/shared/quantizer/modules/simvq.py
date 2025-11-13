# Adapted from https://github.com/lucidrains/vector-quantize-pytorch

from __future__ import annotations

import random
from math import ceil
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from einx import get_at
from torch import Tensor, nn
from torch.nn import Module, ModuleList


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def identity(t):
    return t


def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        (out,) = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse


def rotate_to(x, target):
    x_norm = F.normalize(x, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    cosine_sim = (x_norm * target_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    return x * cosine_sim + target * (1 - cosine_sim)


def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    rand_int = torch.randint(0, max_size, (), device=device)
    if is_distributed():
        dist.all_reduce(rand_int)
    return rand_int.item()


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


class SimVQ(Module):

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        codebook_transform: Optional[Module] = None,
        init_fn: Callable = identity,
        channel_first: bool = True,
        rotation_trick: bool = False,
        input_to_quantize_commit_loss_weight: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.channel_first = channel_first
        frozen_codebook_dim = default(codebook_dim, dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (
            frozen_codebook_dim**-0.5
        )
        codebook = init_fn(codebook)
        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, dim, bias=False)
        self.code_transform = codebook_transform
        self.register_buffer("frozen_codebook", codebook)
        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(self, indices):
        frozen_codes = get_at("[c] d, b ... -> b ... d", self.frozen_codebook, indices)
        quantized = self.code_transform(frozen_codes)
        if self.channel_first:
            quantized = rearrange(quantized, "b ... d -> b d ...")
        return quantized

    def forward(self, x):
        if self.channel_first:
            x = rearrange(x, "b d ... -> b ... d")
        x, inverse_pack = pack_one(x, "b * d")
        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim=-1)
        quantized = get_at("[c] d, b n -> b n d", implicit_codebook, indices)
        commit_loss = (
            F.mse_loss(x.detach(), quantized)
            + F.mse_loss(x, quantized.detach())
            * self.input_to_quantize_commit_loss_weight
        )
        if self.rotation_trick:
            quantized = rotate_to(x, quantized)
        else:
            quantized = (quantized - x).detach() + x
        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, "b *")
        if self.channel_first:
            quantized = rearrange(quantized, "b ... d-> b d ...")
        return quantized, indices, commit_loss
