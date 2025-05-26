#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, Optional, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions


def length_mask(lengths: torch.Tensor, maxlen: int = None) -> torch.Tensor:
    assert lengths.dim() == 1
    maxlen = maxlen if maxlen is not None else lengths.max()
    mask = torch.lt(
        torch.arange(maxlen, device=lengths.device).unsqueeze(0),
        lengths.unsqueeze(1),
    )
    return mask


def causal_mask(qlen: int, device: torch.device) -> torch.Tensor:
    return torch.ones((qlen, qlen), device=device).tril_(0).unsqueeze(0)


def pad_and_concat(tensor_list, pad_id=0):
    """pad a list of torch.Tensor with shape [B_n, T_n, ...]
    in T dimension and then concat in B dimension
    """

    size_list = [t.size() for t in tensor_list]
    concat_size = sum([size[0] for size in size_list])
    pad_size = max([size[1] for size in size_list])
    assert all([size[2:] == size_list[0][2:] for size in size_list])

    retval = (
        torch.ones(
            tuple([concat_size, pad_size] + list(tensor_list[0].size()[2:])),
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )
        * pad_id
    )

    count = 0
    for t in tensor_list:
        B, T = t.size()[:2]
        retval[count : count + B, :T] = t
        count += B

    return retval


def install_kv_cache_hook(model, cache, attn_module):
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
        if isinstance(layer, attn_module):
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks


def logits_to_tokens(
    logits: torch.Tensor,
    opts: SpeechLMInferenceOptions,
    search_algo: str = None,
    allow_eos: bool = True,
    nq_level: int = None,
):
    """
    Select the generated tokens and their scores based on logits prediction.

    logits (torch.Tensor), predicted logits, of size [B, T, nq, V]
    opts (SpeechLMInferenceOptions): search options
    mask (torch.Tensor): mask to specify valid tokens, of size [B, 1, nq, V]
    search_algo (str): search algorithm
    allow_eos (bool): whether to allow end-of-sentence prediction
    nq_level (int or None): if not None, only conpute the specified codec level nq.

    """
    assert logits.dim() == 4
    search_algo = search_algo if search_algo is not None else opts.search_algo
    neg_inf = -1e20

    # (1) Apply mask
    mask = opts.mask.clone()
    mask = mask.view(1, 1, opts.nq, -1)
    if nq_level is not None:
        mask = mask[:, :, nq_level : nq_level + 1]

    if allow_eos:
        mask = mask.clone()
        mask[:, :, 0, opts.eos] = False

    logits.masked_fill_(mask, neg_inf)

    # (2) token selection
    if search_algo in ["topk_sampling"]:
        topk_values, topk_indices = torch.topk(logits, opts.topk, dim=-1)
        probs = torch.softmax(topk_values / opts.sampling_temperature, dim=-1)
        inner_indices = torch.multinomial(
            probs.flatten(end_dim=-2), num_samples=1
        ).view(probs[..., :1].size())
        gen_token_idx = torch.gather(topk_indices, -1, inner_indices).squeeze(-1)
        gen_token_score = torch.gather(probs, -1, inner_indices).squeeze(-1).log()

    elif search_algo in ["topp_sampling"]:
        probs = torch.softmax(logits / opts.sampling_temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        accum_probs = torch.cumsum(sorted_probs, dim=-1)
        clip_probs = torch.where(accum_probs <= opts.top_p, sorted_probs, 0.0)
        # always keep at least one candidate no matter what value it is
        if torch.any(clip_probs[..., 0] == 0.0):
            clip_probs[..., 0] = sorted_probs[..., 0]
        clip_probs = clip_probs / clip_probs.sum(dim=-1, keepdim=True)
        inner_indices = torch.multinomial(
            clip_probs.flatten(end_dim=-2), num_samples=1
        ).view(clip_probs[..., :1].size())
        gen_token_idx = torch.gather(sorted_indices, -1, inner_indices).squeeze(-1)
        gen_token_score = torch.gather(clip_probs, -1, inner_indices).squeeze(-1).log()

    elif search_algo in ["greedy_search", "teacher_force"]:
        probs = logits.softmax(dim=-1)
        topk_values, topk_indices = torch.topk(logits, 1, dim=-1)
        gen_token_idx = topk_indices[:, :, :, 0]
        gen_token_score = topk_values[:, :, :, 0].log()

    else:
        raise NotImplementedError(f"search_algo={search_algo}")

    return gen_token_idx, gen_token_score


def modality_index_to_mask(
    modality_index: torch.Tensor,
    inference_opts: SpeechLMInferenceOptions,
):
    assert modality_index.dim() == 1
    modality_index = modality_index.cpu().tolist()
    mask = torch.stack(
        [inference_opts.masks[idx] for idx in modality_index], dim=0
    ).unsqueeze(
        1
    )  # [B, 1, nq, V]

    return mask


@torch.no_grad()
def install_continuous_features(
    dec_emb: torch.Tensor,
    conti_feats: Tuple = None,
):
    if conti_feats is None:
        return dec_emb

    assert dec_emb.size(0) == len(conti_feats)

    for b, conti_feat in enumerate(conti_feats):
        for conti_emb, start, end in conti_feat:
            # NOTE(Jinchuan): The continuous embeddings shouls have the same dimension
            # of the model hidden size.
            assert conti_emb.size(1) == dec_emb.size(2)
            dec_emb[b, start:end] = conti_emb

    return dec_emb
