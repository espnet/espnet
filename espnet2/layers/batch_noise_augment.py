"""Additive noise augmentation of a padded batch of waveforms, on its device.

The same procedure as ``CommonPreprocessor`` with ``noise_scp`` (per utterance:
a random noise from the list at a random offset, repeated if it is shorter than
the utterance, mixed at an SNR drawn uniformly from ``noise_db_range`` relative
to the power of the non-silent part of the speech, then the result divided by
its peak if that exceeds 1), but done for the whole batch on the device of the
batch, typically the GPU, instead of per utterance on the CPU in the data loader
workers.

All noise recordings are loaded once into one int16 bank in host memory. By
default each step copies only the noise segments of the batch to its device
(2 bytes per sample of the batch). With ``keep_on_device=True`` the whole bank
is copied once to the device instead (2 bytes per sample of the noise list).

The bank is a cache, not a buffer: it is not in the state dict, not broadcast
by DistributedDataParallel, and shared by the replicas of DataParallel. The
module works in training mode only: in eval mode, and when ``noise_apply_prob``
is 0, it returns the input unchanged and loads nothing.
"""

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile
import torch

from espnet2.fileio.read_text import read_2columns_text

# This lock makes the replicas of DataParallel, which run in threads and share
# the cache, load the noise once. It is a module-level object so that the module
# can still be deep-copied.
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
        keep_on_device: Copy the whole noise bank to the device of the batch
            once, instead of copying the noise segments of each batch from host
            memory. Faster, but uses 2 bytes of device memory per noise sample
            (11.4 GB for 99 h at 16 kHz).
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
        keep_on_device: bool = False,
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
        self.keep_on_device = keep_on_device
        # The first training step fills this cache: "host" holds (bank, starts,
        # lengths) on the CPU, and each device key holds a copy on that device.
        # It is a plain dict, not buffers: the replicas of DataParallel share it
        # (they copy the module's attributes shallowly), and
        # DistributedDataParallel does not broadcast it.
        self._cache: Dict[str, Tuple[torch.Tensor, ...]] = {}

    def extra_repr(self) -> str:
        return (
            f"noise_scp={self.noise_scp}, noise_apply_prob={self.noise_apply_prob}, "
            f"noise_db_range=({self.noise_db_low}, {self.noise_db_high}), "
            f"fs={self.fs}, keep_on_device={self.keep_on_device}"
        )

    def _load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks, lengths = [], []
        for path in read_2columns_text(self.noise_scp).values():
            # Read every file as float. Reading a floating-point file as int16
            # does not scale it.
            wav, fs = soundfile.read(path, dtype="float32", always_2d=True)
            if fs != self.fs:
                import librosa

                # Use librosa's default resampler (soxr), which librosa installs.
                wav = librosa.resample(wav.T, orig_sr=fs, target_sr=self.fs).T
            for ch in range(wav.shape[1]):
                x = wav[:, ch]
                if x.size == 0:
                    continue
                # Scale the noise to the full int16 range. The mixing rescales
                # it to the SNR, so its level does not change the result.
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
        """Return (bank int16, starts, lengths) on ``device``.

        The first call loads the noise list; each device gets one copy.
        """
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
        """Return the mean power of the non-silent samples of each row.

        This function computes the same power as detect_non_silence does for the
        utterance alone: it uses the length of each row, not the batch width.

        x: (B, T) float, valid: (B, T) bool (a prefix of each row). Returns (B,).
        """
        b, t = x.shape
        fl, fsh = self.frame_length, self.frame_shift
        length = valid.sum(1)  # (B,)
        x = x * valid
        # Count the frames of each utterance as framing(centered=False,
        # padded=True) does.
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
        # Give sample i the decision of frame i // shift, and the samples after
        # the last frame the decision of the last frame (np.pad(mode="edge")).
        idx = torch.div(torch.arange(t, device=x.device), fsh, rounding_mode="floor")
        idx = torch.minimum(idx[None, :], (n_frames - 1).clamp(min=0)[:, None])
        mask = torch.gather(active, 1, idx)
        # Count every sample of a row that is shorter than one frame or all zero.
        whole = (n_frames == 0)[:, None] | (mean_power == 0)
        mask = torch.where(whole, torch.ones_like(mask), mask) & valid
        return (x.pow(2) * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def _cut_on_host(bank, start, n_len, offset, t, device):
        """Cut the noise segments of a batch from the host bank and copy them.

        Row i is bank[start[i] + (offset[i] + j) % n_len[i]] for j < t. The rows
        are written into one pinned int16 tensor, which is copied to ``device``
        asynchronously. A row whose noise is shorter than the remaining width
        gets one period of the noise (from the offset) and then copies of what
        is already written, doubling each time, so every row is a few memory
        copies.
        """
        pinned = device.type == "cuda"
        out = torch.empty((len(start), t), dtype=torch.int16, pin_memory=pinned)
        bank_np, out_np = bank.numpy(), out.numpy()
        for row, s, n, o in zip(
            out_np, start.tolist(), n_len.tolist(), offset.tolist()
        ):
            noise = bank_np[s : s + n]
            if n - o >= t:
                row[:] = noise[o : o + t]
                continue
            row[: n - o] = noise[o:]
            row[n - o : n] = noise[:o]
            filled = n
            while filled < t:
                c = min(filled, t - filled)
                row[filled : filled + c] = row[:c]
                filled += c
        return out.to(device, non_blocking=pinned)

    def forward(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add noise to speech (B, T) in training mode.

        Returns the speech with noise (same shape and dtype) and speech_lengths.
        """
        if not self.training or self.noise_apply_prob <= 0:
            return speech, speech_lengths
        if speech.dim() != 2:
            raise ValueError(f"speech must be (Batch, Time): {tuple(speech.shape)}")
        b, t = speech.shape
        device = speech.device
        x = speech.float()
        if speech_lengths is None:
            valid = torch.ones(b, t, dtype=torch.bool, device=device)
        else:
            valid = torch.arange(t, device=device) < speech_lengths.to(device)[:, None]

        power = self._non_silent_power(x, valid)

        # Draw a noise and an offset for each utterance: on the device when the
        # bank is there, on the CPU when the noise is cut from host memory.
        bank_device = device if self.keep_on_device else torch.device("cpu")
        bank, starts, lengths = self.noise_bank(bank_device)
        which = torch.randint(len(lengths), (b,), device=bank_device)
        n_len = lengths[which]
        # A long noise starts at any offset where the whole utterance fits
        # (n_len - t + 1 offsets). A short noise starts at any offset and repeats
        # (np.pad(mode="wrap") in the preprocessor).
        span = torch.where(n_len >= t, n_len - t + 1, n_len)
        offset = (torch.rand(b, device=bank_device) * span).long().clamp(max=span - 1)
        if self.keep_on_device:
            pos = (offset[:, None] + torch.arange(t, device=device)) % n_len[:, None]
            noise = bank[starts[which][:, None] + pos]
        else:
            noise = self._cut_on_host(bank, starts[which], n_len, offset, t, device)
        noise = noise.float() / 32768 * valid

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
