#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from espnet2.speechlm.inference_utils import AbsInferenceConfig


def logits_to_tokens(
    logits: torch.Tensor,
    opts: AbsInferenceConfig,
    search_algo: str = None,
    allow_eos: bool = True,
    nq_level: int = None,
):
    """
    Select the generated tokens and their scores based on logits prediction.

    logits (torch.Tensor), predicted logits, of size [B, T, nq, V]
    opts (AbsInferenceConfig): search configurations
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
