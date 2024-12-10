# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from scipy.stats import betabinom


class AlignmentModule(nn.Module):
    """
    Alignment Learning Framework proposed for parallel TTS models in:

    https://arxiv.org/abs/2108.10447

    This module computes the alignment loss between text and acoustic features,
    facilitating effective training of Text-to-Speech (TTS) models using
    attention mechanisms.

    Attributes:
        cache_prior (bool): Whether to cache beta-binomial prior.
        _cache (dict): A cache to store precomputed prior values for efficiency.
        t_conv1 (nn.Conv1d): 1D convolution layer for text features.
        t_conv2 (nn.Conv1d): 1D convolution layer for text features.
        f_conv1 (nn.Conv1d): 1D convolution layer for acoustic features.
        f_conv2 (nn.Conv1d): 1D convolution layer for acoustic features.
        f_conv3 (nn.Conv1d): 1D convolution layer for acoustic features.

    Args:
        adim (int): Dimension of attention.
        odim (int): Dimension of feats.
        cache_prior (bool): Whether to cache beta-binomial prior.

    Examples:
        >>> alignment_module = AlignmentModule(adim=256, odim=80)
        >>> text = torch.randn(4, 10, 256)  # Batch of 4, 10 time steps, adim=256
        >>> feats = torch.randn(4, 20, 80)   # Batch of 4, 20 time steps, odim=80
        >>> text_lengths = torch.tensor([10, 10, 10, 10])
        >>> feats_lengths = torch.tensor([20, 20, 20, 20])
        >>> log_p_attn = alignment_module(text, feats, text_lengths, feats_lengths)
        >>> print(log_p_attn.shape)  # Output shape: (4, 20, 10)
    """

    def __init__(self, adim, odim, cache_prior=True):
        """Initialize AlignmentModule.

        Args:
            adim (int): Dimension of attention.
            odim (int): Dimension of feats.
            cache_prior (bool): Whether to cache beta-binomial prior.

        """
        super().__init__()
        self.cache_prior = cache_prior
        self._cache = {}

        self.t_conv1 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

        self.f_conv1 = nn.Conv1d(odim, adim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

    def forward(self, text, feats, text_lengths, feats_lengths, x_masks=None):
        """
            Calculate alignment loss.

        This method computes the log probability of the attention matrix based on
        the input text embeddings and acoustic features. It performs several
        convolutional operations on the input tensors and computes a score based
        on the Euclidean distance between the features and the text embeddings.
        The resulting score is masked if a mask tensor is provided, and a beta-
        binomial prior is added to the log probabilities before returning the
        final result.

        Args:
            text (Tensor): Batched text embedding (B, T_text, adim).
            feats (Tensor): Batched acoustic feature (B, T_feats, odim).
            text_lengths (Tensor): Text length tensor (B,).
            feats_lengths (Tensor): Feature length tensor (B,).
            x_masks (Tensor, optional): Mask tensor (B, T_text). Defaults to None.

        Returns:
            Tensor: Log probability of attention matrix (B, T_feats, T_text).

        Examples:
            >>> text = torch.randn(2, 5, 256)  # Batch of 2, T_text=5, adim=256
            >>> feats = torch.randn(2, 10, 80)  # Batch of 2, T_feats=10, odim=80
            >>> text_lengths = torch.tensor([5, 3])  # Lengths of each text
            >>> feats_lengths = torch.tensor([10, 8])  # Lengths of each feature
            >>> model = AlignmentModule(adim=256, odim=80)
            >>> log_p_attn = model.forward(text, feats, text_lengths, feats_lengths)
            >>> print(log_p_attn.shape)  # Should print: torch.Size([2, 10, 5])
        """
        text = text.transpose(1, 2)
        text = F.relu(self.t_conv1(text))
        text = self.t_conv2(text)
        text = text.transpose(1, 2)

        feats = feats.transpose(1, 2)
        feats = F.relu(self.f_conv1(feats))
        feats = F.relu(self.f_conv2(feats))
        feats = self.f_conv3(feats)
        feats = feats.transpose(1, 2)

        dist = feats.unsqueeze(2) - text.unsqueeze(1)
        dist = torch.norm(dist, p=2, dim=3)
        score = -dist

        if x_masks is not None:
            x_masks = x_masks.unsqueeze(-2)
            score = score.masked_fill(x_masks, -np.inf)

        log_p_attn = F.log_softmax(score, dim=-1)

        # add beta-binomial prior
        bb_prior = self._generate_prior(
            text_lengths,
            feats_lengths,
        ).to(dtype=log_p_attn.dtype, device=log_p_attn.device)
        log_p_attn = log_p_attn + bb_prior

        return log_p_attn

    def _generate_prior(self, text_lengths, feats_lengths, w=1) -> torch.Tensor:
        """Generate alignment prior formulated as beta-binomial distribution

        Args:
            text_lengths (Tensor): Batch of the lengths of each input (B,).
            feats_lengths (Tensor): Batch of the lengths of each target (B,).
            w (float): Scaling factor; lower -> wider the width.

        Returns:
            Tensor: Batched 2d static prior matrix (B, T_feats, T_text).

        """
        B = len(text_lengths)
        T_text = text_lengths.max()
        T_feats = feats_lengths.max()

        bb_prior = torch.full((B, T_feats, T_text), fill_value=-np.inf)
        for bidx in range(B):
            T = feats_lengths[bidx].item()
            N = text_lengths[bidx].item()

            key = str(T) + "," + str(N)
            if self.cache_prior and key in self._cache:
                prob = self._cache[key]
            else:
                alpha = w * np.arange(1, T + 1, dtype=float)  # (T,)
                beta = w * np.array([T - t + 1 for t in alpha])
                k = np.arange(N)
                batched_k = k[..., None]  # (N,1)
                prob = betabinom.logpmf(batched_k, N, alpha, beta)  # (N,T)

            # store cache
            if self.cache_prior and key not in self._cache:
                self._cache[key] = prob

            prob = torch.from_numpy(prob).transpose(0, 1)  # -> (T,N)
            bb_prior[bidx, :T, :N] = prob

        return bb_prior


@jit(nopython=True)
def _monotonic_alignment_search(log_p_attn):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + log_prob[i, j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 1
        i_b = A[j + 1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a, j] >= Q[i_b, j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    return A


def viterbi_decode(log_p_attn, text_lengths, feats_lengths):
    """
    Extract duration from an attention probability matrix.

    This function computes the token durations from the given log probability
    attention matrix using the Viterbi algorithm. It extracts the most likely
    durations for each token based on the attention probabilities.

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor.

    Examples:
        >>> log_p_attn = torch.randn(2, 5, 10)  # Example attention matrix
        >>> text_lengths = torch.tensor([10, 8])  # Lengths of each text
        >>> feats_lengths = torch.tensor([5, 4])  # Lengths of each feature
        >>> durations, loss = viterbi_decode(log_p_attn, text_lengths, feats_lengths)
        >>> print(durations.shape)  # Should print: torch.Size([2, 10])

    Note:
        The Viterbi algorithm is used to find the most likely sequence of
        states in a hidden Markov model, which in this case relates to the
        alignment between text and features.
    """
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search(cur_log_p_attn.detach().cpu().numpy())
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """
        Average frame-level features into token-level according to durations.

    This function takes in token durations and corresponding feature sequences to
    compute the average feature for each token based on the specified durations.
    It is particularly useful in tasks where features need to be aggregated
    according to their corresponding token lengths, such as in text-to-speech
    applications.

    Args:
        ds (Tensor): Batched token duration (B, T_text).
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched feature averaged according to the token duration
        (B, T_text).

    Examples:
        >>> ds = torch.tensor([[2, 3], [1, 4]])
        >>> xs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
        ...                     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        >>> text_lengths = torch.tensor([2, 2])
        >>> feats_lengths = torch.tensor([5, 8])
        >>> result = average_by_duration(ds, xs, text_lengths, feats_lengths)
        >>> print(result)
        tensor([[2.5000, 4.0000],
                [1.0000, 6.0000]])
    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg
