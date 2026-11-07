"""Additive noise augmentation of a padded batch of waveforms, on its device.

The same procedure as ``CommonPreprocessor`` with ``noise_scp`` (per utterance:
a random noise from the list at a random offset, repeated if it is shorter than
the utterance, mixed at an SNR drawn uniformly from ``noise_db_range`` relative
to the power of the non-silent part of the speech, then the result divided by
its peak if that exceeds 1), but done for the whole batch on the device of the
batch, typically the GPU, instead of per utterance on the CPU in the data loader
workers.

All noise recordings are loaded once into one int16 bank, kept on the host and
copied once to each device that uses it. The bank is a cache, not a buffer: it
is not in the state dict, not broadcast by DistributedDataParallel, and shared
by the replicas of DataParallel. Used in training only: in eval mode, and when
``noise_apply_prob`` is 0, the input is returned unchanged and nothing is loaded.
"""

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile
import torch

from espnet2.fileio.read_text import read_2columns_text

# Serialises the first load: DataParallel runs its replicas in threads, which
# share the cache. Module level, so that the module stays deep-copyable.
_LOAD_LOCK = threading.Lock()


class BatchNoiseAugment(torch.nn.Module):
    """Add noise to a batch of single-channel waveforms on their device.

    Args:
        noise_scp: Kaldi-style list ``<id> <path>`` of noise recordings. Each
            channel of a multi-channel recording is used as a separate noise.
        noise_apply_prob: Probability of adding noise to each utterance.
        noise_db_range: SNR range in dB, ``"low_high"`` or a single value.
        fs: Sampling rate of the speech; noise at another rate is resampled
            when it is loaded.
        silence_threshold: A frame is non-silent when its power exceeds this
            fraction of the mean frame power (``detect_non_silence``).
        frame_length: Frame length (samples) of the silence detection.
        frame_shift: Frame shift (samples) of the silence detection.
    """

    def __init__(
        self,
        noise_scp: str,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "13_15",
        fs: int = 16000,
        silence_threshold: float = 0.01,
        frame_length: int = 1024,
        frame_shift: int = 512,
    ):
        super().__init__()
        self.noise_scp = noise_scp
        self.noise_apply_prob = noise_apply_prob
        bounds = [float(v) for v in str(noise_db_range).split("_")]
        if len(bounds) == 1:
            bounds = bounds * 2
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"noise_db_range must be 'low_high': {noise_db_range}")
        self.noise_db_low, self.noise_db_high = bounds
        self.fs = int(fs)
        self.silence_threshold = silence_threshold
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        # Loaded on the first training step: "host" -> (bank, starts, lengths) on
        # the CPU, and one copy per device. A plain dict, not buffers: the
        # replicas of DataParallel share it (they copy the module's attributes
        # shallowly), and DistributedDataParallel does not broadcast it.
        self._cache: Dict[str, Tuple[torch.Tensor, ...]] = {}

    def extra_repr(self) -> str:
        return (
            f"noise_scp={self.noise_scp}, noise_apply_prob={self.noise_apply_prob}, "
            f"noise_db_range=({self.noise_db_low}, {self.noise_db_high}), fs={self.fs}"
        )

    def _load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks, lengths = [], []
        for path in read_2columns_text(self.noise_scp).values():
            # read as float whatever the file's format (an int16 read of a
            # floating-point file would not be scaled)
            wav, fs = soundfile.read(path, dtype="float32", always_2d=True)
            if fs != self.fs:
                import librosa

                wav = librosa.resample(
                    wav.T, orig_sr=fs, target_sr=self.fs, res_type="kaiser_fast"
                ).T
            for ch in range(wav.shape[1]):
                x = wav[:, ch]
                if x.size == 0:
                    continue
                # the mixing rescales the noise to the SNR, so its level is free:
                # use the int16 range fully
                peak = np.abs(x).max()
                if peak > 0:
                    x = x * (32767.0 / peak)
                chunks.append(np.round(x).astype(np.int16))
                lengths.append(x.size)
        if not chunks:
            raise RuntimeError(f"no noise in {self.noise_scp}")
        lengths = torch.tensor(lengths, dtype=torch.long)
        bank = torch.from_numpy(np.concatenate(chunks))
        starts = torch.cumsum(lengths, 0) - lengths
        logging.info(
            f"BatchNoiseAugment: {len(chunks)} noises, "
            f"{lengths.sum().item() / self.fs / 3600:.1f} h, "
            f"{bank.numel() * 2 / 1e9:.1f} GB"
        )
        return bank, starts, lengths

    def noise_bank(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """(bank int16, starts, lengths) on ``device``, loaded once per device."""
        key = str(device)
        bank = self._cache.get(key)
        if bank is None:
            with _LOAD_LOCK:
                if key not in self._cache:
                    if "host" not in self._cache:
                        self._cache["host"] = self._load()
                    self._cache[key] = tuple(t.to(device) for t in self._cache["host"])
                bank = self._cache[key]
        return bank

    def _non_silent_power(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean power of the non-silent samples of each row, as detect_non_silence.

        Each row is treated as detect_non_silence treats the utterance alone
        (its own length, not the batch width).

        x: (B, T) float, valid: (B, T) bool (a prefix of each row). Returns (B,).
        """
        b, t = x.shape
        fl, fsh = self.frame_length, self.frame_shift
        length = valid.sum(1)  # (B,)
        x = x * valid
        # frames of each utterance alone: framing(..., centered=False, padded=True)
        n_frames = torch.where(
            length >= fl,
            torch.div(length - fl + fsh - 1, fsh, rounding_mode="floor") + 1,
            torch.zeros_like(length),
        )
        f_max = -(-(t - fl) // fsh) + 1 if t >= fl else 1
        padded = torch.nn.functional.pad(x, (0, (f_max - 1) * fsh + fl - t))
        frame_power = padded.unfold(1, fl, fsh).pow(2).mean(-1)[:, :f_max]  # (B, F)
        frame_valid = torch.arange(f_max, device=x.device) < n_frames[:, None]
        mean_power = (frame_power * frame_valid).sum(1, keepdim=True) / n_frames[
            :, None
        ].clamp(min=1)
        active = (frame_power / mean_power.clamp(min=1e-30)) > self.silence_threshold
        # sample i takes the decision of frame i // shift; the tail repeats the
        # last frame (np.pad(mode="edge"))
        idx = torch.div(torch.arange(t, device=x.device), fsh, rounding_mode="floor")
        idx = torch.minimum(idx[None, :], (n_frames - 1).clamp(min=0)[:, None])
        mask = torch.gather(active, 1, idx)
        # shorter than one frame, or all zero: every sample counts
        whole = (n_frames == 0)[:, None] | (mean_power == 0)
        mask = torch.where(whole, torch.ones_like(mask), mask) & valid
        return (x.pow(2) * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add noise to speech (B, T) in training mode.

        Returns the augmented speech (same shape and dtype) and speech_lengths.
        """
        if not self.training or self.noise_apply_prob <= 0:
            return speech, speech_lengths
        if speech.dim() != 2:
            raise ValueError(f"speech must be (Batch, Time): {tuple(speech.shape)}")
        bank, starts, lengths = self.noise_bank(speech.device)

        b, t = speech.shape
        device = speech.device
        x = speech.float()
        if speech_lengths is None:
            valid = torch.ones(b, t, dtype=torch.bool, device=device)
        else:
            valid = torch.arange(t, device=device) < speech_lengths.to(device)[:, None]

        power = self._non_silent_power(x, valid)

        # a noise, an offset and an SNR per utterance
        which = torch.randint(len(lengths), (b,), device=device)
        n_len = lengths[which]
        # long noise: any offset where the whole utterance fits (n_len - t + 1);
        # short noise: any offset, repeated (np.pad(mode="wrap") in the preprocessor)
        span = torch.where(n_len >= t, n_len - t + 1, n_len)
        offset = (torch.rand(b, device=device) * span).long().clamp(max=span - 1)
        pos = (offset[:, None] + torch.arange(t, device=device)) % n_len[:, None]
        noise = bank[starts[which][:, None] + pos].float() / 32768
        noise = noise * valid

        n = valid.sum(1).clamp(min=1)
        noise_power = noise.pow(2).sum(1) / n
        snr = torch.empty(b, device=device).uniform_(
            self.noise_db_low, self.noise_db_high
        )
        scale = 10 ** (-snr / 20) * power.sqrt() / noise_power.clamp(min=1e-10).sqrt()
        apply = torch.rand(b, device=device) < self.noise_apply_prob
        x = x + (scale * apply)[:, None] * noise

        # as the preprocessor: divide by the peak when it exceeds 1
        peak = (x.abs() * valid).amax(1, keepdim=True)
        x = torch.where(peak > 1.0, x / peak.clamp(min=1e-30), x)
        return x.to(speech.dtype), speech_lengths
