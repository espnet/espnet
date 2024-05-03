#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict

import torch
from espnet2.speechlm.module.module import MultiHeadedAttention


def length_mask(lengths: torch.Tensor) -> torch.Tensor:
    assert lengths.dim() == 1
    mask = torch.le(
        torch.arange(lengths.max(), device=lengths.device).unsqueeze(0),
        lengths.unsqueeze(1),
    ).long()
    return mask

def causal_mask(qlen: int, device: torch.device) -> torch.Tensor:
    return torch.ones((qlen, qlen), device=device).tril_(0).unsqueeze(0)

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
        if isinstance(layer, MultiHeadedAttention):
            hooks.append(layer.linear_k.register_forward_hook(save_to_cache))
            hooks.append(layer.linear_v.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks

def ce_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
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
        elem_loss = elem_loss * mask
        loss = elem_loss.sum() / mask.sum() / logits.size(2)

        pred = logits.argmax(dim=-1)
        acc = torch.eq(pred, target).to(elem_loss.dtype) * mask
        
        stats = {}
        for nq_idx in range(target.size(2)):
            stats.update({
                f"acc_layer{nq_idx}": acc[:, :, nq_idx].sum() / mask.sum()
            })
        
        acc = acc.sum() / mask.sum() / logits.size(2)
        stats.update({"loss": loss.clone().detach(), "acc": acc})
        weight = mask.sum()

        return loss, stats, weight
