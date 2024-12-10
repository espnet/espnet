#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.module.transformer import MultiHeadAttention


def length_mask(lengths: torch.Tensor, maxlen: int = None) -> torch.Tensor:
    """
    Creates a length mask tensor based on input lengths.

    The length mask is a binary tensor that indicates which positions
    in a sequence are valid based on the given lengths. This is useful
    for masking out padding tokens in batch processing of variable-length
    sequences.

    Args:
        lengths (torch.Tensor): A 1D tensor containing the lengths of each
            sequence in the batch.
        maxlen (int, optional): The maximum length of sequences. If not
            provided, it defaults to the maximum value in `lengths`.

    Returns:
        torch.Tensor: A 2D binary tensor of shape (num_sequences, maxlen),
        where each row contains 1s for valid positions and 0s for
        positions beyond the length of the corresponding sequence.

    Examples:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = length_mask(lengths)
        >>> print(mask)
        tensor([[1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0]])

    Note:
        The output mask is generated using broadcasting to efficiently
        create the desired shape.

    Raises:
        AssertionError: If `lengths` is not a 1D tensor.
    """
    assert lengths.dim() == 1
    maxlen = maxlen if maxlen is not None else lengths.max()
    mask = torch.lt(
        torch.arange(maxlen, device=lengths.device).unsqueeze(0),
        lengths.unsqueeze(1),
    ).long()
    return mask


def causal_mask(qlen: int, device: torch.device) -> torch.Tensor:
    """
        Generates a causal mask tensor for attention mechanisms.

    A causal mask is a square matrix that allows attention to only consider
    previous tokens in a sequence. This is particularly useful in autoregressive
    models where the prediction of a token should not depend on future tokens.

    Args:
        qlen (int): The length of the sequence for which to create the mask.
        device (torch.device): The device (CPU or GPU) on which the mask will be
            allocated.

    Returns:
        torch.Tensor: A tensor of shape (1, qlen, qlen) representing the causal
        mask, where the lower triangular part is filled with ones and the upper
        triangular part is filled with zeros.

    Examples:
        >>> mask = causal_mask(5, torch.device('cpu'))
        >>> print(mask)
        tensor([[[1., 0., 0., 0., 0.],
                 [1., 1., 0., 0., 0.],
                 [1., 1., 1., 0., 0.],
                 [1., 1., 1., 1., 0.],
                 [1., 1., 1., 1., 1.]]])
    """
    return torch.ones((qlen, qlen), device=device).tril_(0).unsqueeze(0)


def ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    prefix_len: torch.Tensor = None,
    first_layer_weight: int = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Computes the cross-entropy loss and accuracy for the given logits and targets.

    This function calculates the cross-entropy loss between the predicted logits and the
    target values, applying a length mask to ignore padding tokens. It also computes
    accuracy for each layer in the output.

    Attributes:
        logits (torch.Tensor): The predicted logits of shape (B, T, N, C) where B is the
            batch size, T is the sequence length, N is the number of layers, and C is
            the number of classes.
        target (torch.Tensor): The target tensor of shape (B, T, N) containing the
            correct class indices for each token.
        lengths (torch.Tensor): A tensor of shape (B,) indicating the actual lengths
            of each sequence in the batch.
        prefix_len (torch.Tensor, optional): A tensor indicating the lengths of the
            prefixes to mask out in the loss computation. Defaults to None.
        first_layer_weight (float, optional): Weight to apply to the first layer's
            gradients. Defaults to 1.0.

    Args:
        logits (torch.Tensor): Predicted logits from the model.
        target (torch.Tensor): Ground truth target values.
        lengths (torch.Tensor): Lengths of the sequences for masking.
        prefix_len (torch.Tensor, optional): Prefix lengths for masking.
        first_layer_weight (float, optional): Weight for the first layer.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple containing
            the computed loss, a dictionary with accuracy statistics for each layer,
            and the total weight used in loss computation.

    Raises:
        AssertionError: If the dimensions of logits or the sizes do not match the target.

    Examples:
        >>> logits = torch.randn(2, 5, 3, 10)  # Example logits
        >>> target = torch.randint(0, 10, (2, 5, 3))  # Example target
        >>> lengths = torch.tensor([5, 3])  # Actual lengths
        >>> loss, stats, weight = ce_loss(logits, target, lengths)

    Note:
        This function assumes that the input tensors are correctly shaped and
        that the required dimensions are as specified.
    """
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
    """
        Install key-value cache hooks for the specified model layers.

    This function registers hooks on the `MultiHeadAttention` layers of the
    provided model. The hooks will save the key-value outputs to a cache
    dictionary during the forward pass, allowing for efficient reuse of
    these outputs in subsequent steps, which is particularly useful for
    transformer models during autoregressive generation.

    Args:
        model (torch.nn.Module): The model containing the layers on which to
            install the hooks.
        cache (dict, optional): A dictionary to store the cached key-value
            outputs. If None, an empty dictionary will be created.

    Returns:
        Tuple[dict, list]: A tuple containing:
            - cache (dict): The updated cache dictionary with stored key-value
              outputs.
            - hooks (list): A list of the registered hooks for cleanup if needed.

    Examples:
        >>> from espnet2.speechlm.module.transformer import TransformerModel
        >>> model = TransformerModel(...)
        >>> cache = {}
        >>> cache, hooks = install_kv_cache_hook(model, cache)

    Note:
        Make sure to clean up the hooks after use to prevent memory leaks by
        calling `remove()` on each hook in the `hooks` list.

    Todo:
        - Implement automatic cleanup of hooks if the model is deleted.
    """
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
    """
        Converts logits to tokens based on specified inference options and search
    algorithm.

    This function takes a tensor of logits and applies a mask based on the
    inference options. It then selects tokens from the logits using the specified
    search algorithm, which can be either sampling or greedy search.

    Args:
        logits (torch.Tensor): A 4D tensor of shape (batch_size, num_heads,
            sequence_length, vocab_size) representing the model's output logits.
        opts (SpeechLMInferenceOptions): An object containing inference options,
            including masks, top_k, eos token index, and search algorithm.
        allow_eos (bool, optional): Whether to allow the End Of Sequence (EOS)
            token to be predicted. Defaults to True.
        nq_level (int, optional): The specific level for token generation. If
            None, it uses the default level. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - gen_token_idx (torch.Tensor): A tensor of generated token indices
                of shape (1, batch_size, num_heads, sequence_length).
            - gen_token_score (torch.Tensor): A tensor of scores corresponding to
                the generated tokens, of shape (1, batch_size, num_heads,
                sequence_length).

    Raises:
        NotImplementedError: If the specified search algorithm is not implemented.

    Examples:
        >>> from espnet2.speechlm.core_lm import SpeechLMInferenceOptions
        >>> logits = torch.randn(2, 4, 10, 100)  # Example logits
        >>> opts = SpeechLMInferenceOptions(masks=torch.ones(1, 1, 10, 100),
        ...                                  eos=99, top_k=5,
        ...                                  search_algo='greedy_search')
        >>> gen_token_idx, gen_token_score = logits_to_tokens(logits, opts)
        >>> print(gen_token_idx.shape)  # Output: (1, 2, 4, 10)
        >>> print(gen_token_score.shape)  # Output: (1, 2, 4, 10)
    """
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
