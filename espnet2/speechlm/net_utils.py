#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.module.transformer import MultiHeadAttention


def length_mask(lengths: torch.Tensor, maxlen: int = None) -> torch.Tensor:
    assert lengths.dim() == 1
    maxlen = maxlen if maxlen is not None else lengths.max()
    mask = torch.lt(
        torch.arange(maxlen, device=lengths.device).unsqueeze(0),
        lengths.unsqueeze(1),
    ).long()
    return mask


def causal_mask(qlen: int, device: torch.device) -> torch.Tensor:
    return torch.ones((qlen, qlen), device=device).tril_(0).unsqueeze(0)


def ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    prefix_len: torch.Tensor = None,
    first_layer_weight: int = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert logits.dim() == 4
    assert logits.size()[:3] == target.size()

    if first_layer_weight != 1.0 and logits.requires_grad:

        def hook(grad):
            grad[:, :, 0, :] *= first_layer_weight
            return grad

        logits.register_hook(hook)

    elem_loss = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2), target, reduction="none"
    )

    mask = length_mask(lengths).to(elem_loss.dtype).unsqueeze(-1)
    if prefix_len is not None:
        target_mask = (
            length_mask(prefix_len, maxlen=lengths.max())
            .to(elem_loss.dtype)
            .unsqueeze(-1)
        )
        target_mask = mask * torch.abs(target_mask - 1)
    else:
        target_mask = mask

    # compute loss on each token
    elem_loss = elem_loss * mask
    loss = elem_loss.sum() / mask.sum() / logits.size(2)

    # compute accuracy only on target tokens only
    pred = logits.argmax(dim=-1)
    acc = torch.eq(pred, target).to(elem_loss.dtype) * target_mask

    stats = {}
    for nq_idx in range(target.size(2)):
        stats.update(
            {f"acc_layer{nq_idx}": acc[:, :, nq_idx].sum() / target_mask.sum()}
        )

    acc = acc.sum() / target_mask.sum() / logits.size(2)
    stats.update({"loss": loss.clone().detach(), "acc": acc})
    weight = mask.sum()

    return loss, stats, weight


def install_kv_cache_hook(model, cache):
    cache = {**cache} if cache is not None else {}
    hooks = []

    def save_to_cache(module, _, output):
        if module not in cache:
            # save as-is, for the first token or cross attention
            cache[module] = output
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def install_hooks(layer: torch.nn.Module):
        if isinstance(layer, MultiHeadAttention):
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks


def logits_to_tokens(
    logits: torch.Tensor,
    opts: SpeechLMInferenceOptions,
    allow_eos: bool = True,
    nq_level: int = None,
):
    assert logits.dim() == 4

    # (1) Apply mask
    mask = opts.masks
    if allow_eos:  # only predict eos in the first code
        mask[..., 0, opts.eos] = False
    if nq_level is not None:
        mask = mask[nq_level : nq_level + 1]
    mask = mask.unsqueeze(0).unsqueeze(0)
    logits = logits.masked_fill_(mask, -1e20)

    # (2) token selection
    topk_values, topk_indices = torch.topk(logits, opts.top_k, dim=-1)

    if opts.search_algo in ["sampling"]:
        logp = torch.softmax(topk_values / opts.sampling_temperature, dim=-1)
        inner_indices = torch.multinomial(logp.flatten(end_dim=-2), num_samples=1).view(
            logp[..., :1].size()
        )
        gen_token_idx = torch.gather(topk_indices, -1, inner_indices).squeeze(-1)
        gen_token_score = torch.gather(topk_values, -1, inner_indices).squeeze(-1)

    elif opts.search_algo in ["greedy_search", "teacher_force"]:
        gen_token_idx = topk_indices[:, :, :, 0]
        gen_token_score = topk_values[:, :, :, 0]

    else:
        raise NotImplementedError(f"opts.search_algo={opts.search_algo}")

    return gen_token_idx, gen_token_score
