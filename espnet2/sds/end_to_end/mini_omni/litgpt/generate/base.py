# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import Any, Optional

import torch
from tqdm import tqdm

from espnet2.sds.end_to_end.mini_omni.litgpt.model import GPT
from espnet2.sds.end_to_end.mini_omni.utils.snac_utils import layershift, snac_config

# import torch._dynamo.config
# import torch._inductor.config


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs,
        # num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] ->
    # sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with
        # a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(
    model: GPT, input_pos: torch.Tensor, x: list, **kwargs: Any
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    logits_a, logit_t = model(x, input_pos)

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=x[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=x[0].dtype)
    return next_audio_tokens, next_t


def next_token_asr(
    model: GPT,
    input_pos: torch.Tensor,
    audio_features: torch.tensor,
    lens: int,
    input_ids: list,
    **kwargs: Any,
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    logits_a, logit_t = model(audio_features, input_ids, input_pos, whisper_lens=lens)

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=input_ids[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_audio_tokens, next_t


def next_token_A1T2(
    model: GPT,
    audio_features: torch.tensor,
    input_ids: list,
    whisper_lens: int,
    task: list,
    input_pos: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    logits_a, logit_t = model(
        audio_features, input_ids, input_pos, whisper_lens=whisper_lens, task=task
    )

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=input_ids[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_audio_tokens, next_t


def next_token_A1T1(
    model: GPT,
    audio_features: torch.tensor,
    input_ids: list,
    whisper_lens: int,
    task: list,
    input_pos: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    logits_a, logit_t = model(
        audio_features, input_ids, input_pos, whisper_lens=whisper_lens, task=task
    )
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_t


def next_token_batch(
    model: GPT,
    audio_features: torch.tensor,
    input_ids: list,
    whisper_lens: int,
    task: list,
    input_pos: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    input_pos = input_pos.to(model.device)
    input_ids = [input_id.to(model.device) for input_id in input_ids]
    logits_a, logit_t = model(
        audio_features, input_ids, input_pos, whisper_lens=whisper_lens, task=task
    )

    for i in range(7):
        logits_a[i] = logits_a[i][0].unsqueeze(0)
    logit_t = logit_t[1].unsqueeze(0)

    next_audio_tokens = []
    for logit_a in logits_a:
        next_a = sample(logit_a, **kwargs).to(dtype=input_ids[0].dtype)
        next_audio_tokens.append(next_a)
    next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    return next_audio_tokens, next_t


# torch._dynamo.config.automatic_dynamic_shapes = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.coordinate_descent_tuning = True
# next_token = torch.compile(next_token, mode="reduce-overhead")


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: list,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:
    # print("eos_id_a:", eos_id_a)
    # print("eos_id_t:", eos_id_t)
    # print("pad_id:", pad_id)
    """Takes a conditioning sequence (prompt) as input and continues to

    generate as many tokens as requested.
    The implementation of this function is modified from A.
    Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return
         (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k
        highest probabilities.
        top_p: If specified, it represents the cumulative
        probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the
            highest probability tokens
            whose cumulative probability exceeds the threshold
            `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is
            equivalent
            to sampling the most probable token, while `top_p=1`
            samples from the whole distribution.
            It can be used in conjunction with `top_k` and
            `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        eos_id: If specified, stop generating any more token once
        the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after
        applying the prompt style) to the output.
    """
    T = input_ids[0].size(0)
    device = input_ids[0].device
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would
        # be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model
        # compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall
        # speed
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to be >="
            f"{max_returned_tokens - 1}"
        )

    for input_id in input_ids:
        input_id = [input_id]
    (
        tokens_A1,
        tokens_A2,
        tokens_A3,
        tokens_A4,
        tokens_A5,
        tokens_A6,
        tokens_A7,
        tokens_T,
    ) = input_ids

    tokens_A1_output = [tokens_A1]
    tokens_A2_output = [tokens_A2]
    tokens_A3_output = [tokens_A3]
    tokens_A4_output = [tokens_A4]
    tokens_A5_output = [tokens_A5]
    tokens_A6_output = [tokens_A6]
    tokens_A7_output = [tokens_A7]
    tokens_T_output = [tokens_T]

    list_output = [
        tokens_A1_output,
        tokens_A2_output,
        tokens_A3_output,
        tokens_A4_output,
        tokens_A5_output,
        tokens_A6_output,
        tokens_A7_output,
        tokens_T_output,
    ]

    input_pos = torch.tensor([T], device=device)
    model_input_ids = [
        tokens_A1.view(1, -1),
        tokens_A2.view(1, -1),
        tokens_A3.view(1, -1),
        tokens_A4.view(1, -1),
        tokens_A5.view(1, -1),
        tokens_A6.view(1, -1),
        tokens_A7.view(1, -1),
        tokens_T.view(1, -1),
    ]

    tokens_A, token_T = next_token(
        model,
        torch.arange(0, T, device=device),
        model_input_ids,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    for i in range(7):
        list_output[i].append(tokens_A[i].clone())
    list_output[7].append(token_T.clone())

    # prepare the input for the next iteration
    for i in range(7):
        tokens_A[i] = tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
    token_T = token_T.clone()

    text_end = False
    max_returned_tokens = 1000
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        model_input_ids = [
            token_a.view(1, -1).to(torch.int32) for token_a in tokens_A
        ] + [token_T.view(1, -1).to(torch.int32)]
        tokens_A, token_T = next_token(
            model,
            input_pos,
            model_input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if text_end:
            token_T = torch.tensor([pad_id], device=device)

        for i in range(7):
            list_output[i].append(tokens_A[i].clone())
        list_output[7].append(token_T.clone())

        if tokens_A[-1] == eos_id_a:
            break
        if token_T == eos_id_t:
            if generate_text:
                break
            text_end = True

        for i in range(7):
            tokens_A[i] = (
                tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
            )
        token_T = token_T.clone()
        input_pos = input_pos.add_(1)

    for i in range(len(list_output)):
        list_output[i] = torch.cat(list_output[i])
    return list_output


@torch.inference_mode()
def generate_TA_BATCH(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 1000,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:

    T = input_ids[0].size(1)
    device = input_ids[0].device
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        raise NotImplementedError(
            f"max_seq_length {model.max_seq_length} needs to "
            f"be >= {max_returned_tokens - 1}"
        )

    input_pos = torch.tensor([T], device=device)
    model_input_ids = input_ids

    list_output = [[] for i in range(8)]

    tokens_A, token_T = next_token_batch(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3, T - 3],
        ["A1T2", "A1T2"],
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    for i in range(7):
        list_output[i].append(tokens_A[i].tolist()[0])
    list_output[7].append(token_T.tolist()[0])

    model_input_ids = [[] for i in range(8)]
    for i in range(7):
        tokens_A[i] = tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
        model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
        model_input_ids[i].append(
            torch.tensor([layershift(snac_config.end_of_audio, i)], device=device)
        )
        model_input_ids[i] = torch.stack(model_input_ids[i])

    model_input_ids[-1].append(token_T.clone().to(torch.int32))
    model_input_ids[-1].append(token_T.clone().to(torch.int32))
    model_input_ids[-1] = torch.stack(model_input_ids[-1])

    text_end = False

    for _ in range(2, max_returned_tokens - T + 1):
        tokens_A, token_T = next_token_batch(
            model,
            None,
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if text_end:
            token_T = torch.tensor([pad_id_t], device=device)

        if tokens_A[-1] == eos_id_a:
            break
        if token_T == eos_id_t:
            text_end = True

        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            tokens_A[i] = (
                tokens_A[i].clone() + shift + i * snac_config.padded_vocab_size
            )
            model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
            model_input_ids[i].append(
                torch.tensor([layershift(snac_config.end_of_audio, i)], device=device)
            )
            model_input_ids[i] = torch.stack(model_input_ids[i])

        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        input_pos = input_pos.add_(1)

    return list_output


@torch.inference_mode()
def generate_TT(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:

    T = input_ids[0].size(1)
    device = input_ids[0].device

    output = []
    token_T = next_token_A1T1(
        model,
        None,
        input_ids,
        None,
        None,
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    output.append(token_T.clone().tolist()[0])
    input_pos = torch.tensor([T], device=device)

    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        model_input_ids = []
        for i in range(7):
            model_input_ids.append(
                torch.tensor([layershift(snac_config.end_of_audio, i)])
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))
        token_T = next_token_A1T1(
            model,
            None,
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if token_T == eos_id_t:
            break
        output.append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)
    return output


@torch.inference_mode()
def generate_AT(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:

    T = input_ids[0].size(1)
    device = input_ids[0].device

    output = []
    token_T = next_token_A1T1(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3],
        ["AT"],
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    output.append(token_T.clone().tolist()[0])
    input_pos = torch.tensor([T], device=device)
    text_end = False  # noqa
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        model_input_ids = []
        for i in range(7):
            model_input_ids.append(
                torch.tensor([layershift(snac_config.end_of_audio, i)])
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))
        token_T = next_token_A1T1(
            model,
            None,
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if token_T == eos_id_t:
            break
        output.append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)
    return output


@torch.inference_mode()
def generate_TA(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:

    T = input_ids[0].size(1)
    device = input_ids[0].device

    output = [[] for _ in range(8)]
    tokens_A, token_T = next_token_A1T2(
        model,
        None,
        input_ids,
        None,
        None,
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    for i in range(7):
        output[i].append(tokens_A[i].clone().tolist()[0])
    output[7].append(token_T.clone().tolist()[0])

    input_pos = torch.tensor([T], device=device)
    text_end = False
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):

        model_input_ids = []
        for i in range(7):
            model_input_ids.append(
                layershift(tokens_A[i].clone(), i)
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))

        tokens_A, token_T = next_token_A1T2(
            model,
            None,
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if text_end:
            token_T = torch.tensor([pad_id_t], device=device)

        if tokens_A[-1] == eos_id_a:
            break

        if token_T == eos_id_t:
            text_end = True

        for i in range(7):
            output[i].append(tokens_A[i].clone().tolist()[0])
        output[7].append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)

    return output


@torch.inference_mode()
def generate_AA(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:

    T = input_ids[0].size(1)
    device = input_ids[0].device

    output = [[] for _ in range(8)]
    tokens_A, token_T = next_token_A1T2(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3],
        ["A1T2"],
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    for i in range(7):
        output[i].append(tokens_A[i].clone().tolist()[0])
    output[7].append(token_T.clone().tolist()[0])

    input_pos = torch.tensor([T], device=device)

    text_end = False
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):

        model_input_ids = []
        for i in range(7):
            model_input_ids.append(
                layershift(tokens_A[i].clone(), i)
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))

        tokens_A, token_T = next_token_A1T2(
            model,
            None,
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if text_end:
            token_T = torch.tensor([pad_id_t], device=device)

        if tokens_A[-1] == eos_id_a:
            break
        if token_T == eos_id_t:
            # print("text_end")
            text_end = True

        for i in range(7):
            output[i].append(tokens_A[i].clone().tolist()[0])
        output[7].append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)

    return output


@torch.inference_mode()
def generate_ASR(
    model: GPT,
    audio_features: torch.Tensor,
    input_ids: list,
    leng,
    task,
    max_returned_tokens: int = 1200,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> torch.Tensor:

    T = input_ids[0].size(1)
    device = input_ids[0].device
    output = []
    token_T = next_token_A1T1(
        model,
        audio_features.to(torch.float32).to(model.device),
        input_ids,
        [T - 3],
        ["asr"],
        input_pos=torch.arange(0, T, device=device),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    output.append(token_T.clone().tolist()[0])
    input_pos = torch.tensor([T], device=device)
    text_end = False  # noqa
    for _ in tqdm(range(2, max_returned_tokens - T + 1)):
        model_input_ids = []
        for i in range(7):
            model_input_ids.append(
                torch.tensor([layershift(snac_config.end_of_audio, i)])
                .view(1, -1)
                .to(torch.int32)
                .to(device)
            )
        model_input_ids.append(token_T.clone().view(1, -1).to(torch.int32).to(device))
        token_T = next_token_A1T1(
            model,
            None,
            model_input_ids,
            None,
            None,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if token_T == eos_id_t:
            break
        output.append(token_T.clone().tolist()[0])
        input_pos = input_pos.add_(1)
    return output
