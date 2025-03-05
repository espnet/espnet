import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
from einops import rearrange, repeat


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    logging.info(
        f"Running K-means with {num_clusters} clusters, {num_iters} iterations, and cosine similarity: {use_cosine_sim}"
    )

    means = sample_vectors(samples, num_clusters)
    logging.info(f"Init means!")

    for _ in range(num_iters):
        logging.info(f"Running iteration {_ + 1}...")
        if use_cosine_sim:
            # Assumes samples are normalized
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, "n d -> n () d") - rearrange(
                means, "c d -> () c d"
            )
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


@torch.no_grad()
def beats_frontend(
    source: torch.Tensor,
    fbank_mean: float,
    fbank_std: float,
) -> torch.Tensor:
    """Preprocess raw audio."""
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2**15  # float32 to int16
        fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=128,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
        )
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank


@torch.no_grad()
def forward_padding_mask_conv(
    padding_mask: torch.Tensor,
    n_dim: int,
    conv_module: nn.Module,
):
    """Forward padding mask.
    To be applied after features are passed through conv module or after converting to spectrogram,
    for consistency.
    padding_mask: BT
    n_dim: number of dimensions before the transformation was applied to features.
        When applying after fbank computation set this to a non-positive value.
    conv_module: conv module applied to features,
        the channel dimension must be 1.
    """
    assert padding_mask.dim() == 2
    if n_dim >= 1:
        padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, n_dim)  # btn
    padding_mask = padding_mask.unsqueeze(1)  # b1tn or b1t, depending on n_dim
    dtype_ = next(conv_module.parameters()).dtype
    padding_mask = conv_module(padding_mask.to(dtype_))
    padding_mask = padding_mask != 0
    padding_mask = (
        padding_mask.view(padding_mask.shape[0], padding_mask.shape[1], -1)
        .squeeze(-2)
        .contiguous()
    )
    return padding_mask


def freeze_conv_module(conv_module: nn.Module):
    # Fix patch embedding for padding
    conv_module.weight.data.fill_(1)
    conv_module.weight.requires_grad = False
    if conv_module.bias is not None:
        conv_module.bias.data.fill_(0)
        conv_module.bias.requires_grad = False
