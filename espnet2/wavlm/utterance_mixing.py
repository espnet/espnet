#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# WavLM utterance mixing / speech denoising.
#     Paper: https://arxiv.org/abs/2110.13900 (Sec. 2.2, "masked speech denoising
#     and prediction")
#
# During pretraining each primary utterance is, with some probability, mixed
# with a segment of another (secondary) utterance drawn from the same batch. The
# model sees the corrupted mixture as input but must still predict the k-means
# cluster targets of the *clean primary* speech -- forcing it to denoise /
# separate the dominant speaker. Targets are therefore left untouched; only the
# waveform is modified, and only during training.

from typing import Tuple

import torch
from torch import nn


class UtteranceMixing(nn.Module):
    """In-batch utterance mixing for WavLM-style speech denoising.

    Args:
        mixing_prob: Probability that a given primary utterance is mixed.
        max_num_sources: Max number of secondary utterances mixed into one
            primary (each is drawn and added independently).
        snr_low: Lowest SNR (dB) of primary-over-secondary at mixing.
        snr_high: Highest SNR (dB) of primary-over-secondary at mixing.
        max_len_ratio: Upper bound on the interfering segment length as a
            fraction of the primary's valid length.
    """

    def __init__(
        self,
        mixing_prob: float = 0.2,
        max_num_sources: int = 1,
        snr_low: float = -5.0,
        snr_high: float = 20.0,
        max_len_ratio: float = 0.5,
    ):
        super().__init__()
        assert 0.0 <= mixing_prob <= 1.0, mixing_prob
        assert max_num_sources >= 1, max_num_sources
        assert snr_low <= snr_high, (snr_low, snr_high)
        assert 0.0 < max_len_ratio <= 1.0, max_len_ratio
        self.mixing_prob = mixing_prob
        self.max_num_sources = max_num_sources
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.max_len_ratio = max_len_ratio

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix each primary utterance with in-batch secondary segments.

        Args:
            speech: (Batch, NSamples) raw waveform.
            speech_lengths: (Batch,) number of valid samples per utterance.

        Returns:
            (mixed_speech, speech_lengths) -- lengths are unchanged.
        """
        batch_size = speech.shape[0]
        # Need at least two utterances to draw an interferer.
        if batch_size < 2 or self.mixing_prob <= 0.0:
            return speech, speech_lengths

        device = speech.device
        # Clone so we read from the original (clean) batch while writing mixes.
        clean = speech
        mixed = speech.clone()

        for i in range(batch_size):
            if torch.rand(1, device=device).item() >= self.mixing_prob:
                continue

            len_i = int(speech_lengths[i])
            if len_i <= 1:
                continue

            # power of the primary's valid region (for SNR scaling)
            primary = clean[i, :len_i]
            primary_power = primary.pow(2).mean().clamp_min(1e-10)

            num_sources = int(
                torch.randint(1, self.max_num_sources + 1, (1,), device=device).item()
            )

            for _ in range(num_sources):
                # pick a secondary utterance j != i
                j = int(torch.randint(0, batch_size, (1,), device=device).item())
                if j == i:
                    j = (j + 1) % batch_size
                len_j = int(speech_lengths[j])
                if len_j <= 1:
                    continue

                # interfering segment length: up to max_len_ratio of primary,
                # and no longer than the secondary's valid length
                max_seg = max(1, int(self.max_len_ratio * len_i))
                seg_len = min(max_seg, len_j)
                if seg_len <= 0:
                    continue

                # random crop position within the secondary
                sec_start = int(
                    torch.randint(0, len_j - seg_len + 1, (1,), device=device).item()
                )
                secondary = clean[j, sec_start : sec_start + seg_len]

                # scale the interferer to a random SNR relative to the primary
                snr_db = (
                    torch.rand(1, device=device).item()
                    * (self.snr_high - self.snr_low)
                    + self.snr_low
                )
                sec_power = secondary.pow(2).mean().clamp_min(1e-10)
                # primary_power / (scale^2 * sec_power) = 10^(snr/10)
                scale = torch.sqrt(
                    primary_power / (sec_power * (10.0 ** (snr_db / 10.0)))
                )

                # random start offset inside the primary's valid region
                dst_start = int(
                    torch.randint(0, len_i - seg_len + 1, (1,), device=device).item()
                )
                mixed[i, dst_start : dst_start + seg_len] = (
                    mixed[i, dst_start : dst_start + seg_len] + scale * secondary
                )

        return mixed, speech_lengths
